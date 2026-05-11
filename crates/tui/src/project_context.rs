//! Project context loading for DeepSeek TUI.
//!
//! This module handles loading project-specific context files that provide
//! instructions and context to the AI agent. These include:
//!
//! - `AGENTS.md` - Project-level agent instructions (primary)
//! - `.claude/instructions.md` - Claude-style hidden instructions
//! - `CLAUDE.md` - Claude-style instructions
//! - `.deepseek/instructions.md` - Hidden instructions file (legacy)
//!
//! The loaded content is injected into the system prompt to give the agent
//! context about the project's conventions, structure, and requirements.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ignore::{DirEntry, WalkBuilder};
use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Names of project context files to look for, in priority order.
const PROJECT_CONTEXT_FILES: &[&str] = &[
    "AGENTS.md",
    ".claude/instructions.md",
    "CLAUDE.md",
    ".deepseek/instructions.md",
];

/// User-level project instructions loaded as a fallback when the workspace and
/// its parents do not define project context.
const GLOBAL_AGENTS_RELATIVE_PATH: &[&str] = &[".deepseek", "AGENTS.md"];

/// Maximum size for project context files (to prevent loading huge files)
const MAX_CONTEXT_SIZE: usize = 100 * 1024; // 100KB
const PACK_README_MAX_CHARS: usize = 4_000;
const PACK_MAX_ENTRIES: usize = 400;
const PACK_MAX_SOURCE_FILES: usize = 80;
const PACK_MAX_CONFIG_FILES: usize = 80;
const PACK_MAX_DEPTH: usize = 4;
const PACK_IGNORED_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".next",
    ".cache",
    "coverage",
    "logs",
    "tmp",
    "temp",
    ".tmp",
    ".idea",
    ".vscode",
    ".pytest_cache",
];
const PACK_IGNORED_FILES: &[&str] = &[".ds_store", "thumbs.db"];

#[derive(Debug, Clone)]
struct CachedProjectPack {
    manifest_hash: String,
    rendered: String,
}

static PROJECT_PACK_CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedProjectPack>>> = OnceLock::new();

// === Errors ===

#[derive(Debug, Error)]
enum ProjectContextError {
    #[error("Failed to read context metadata for {path}: {source}")]
    Metadata {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Context file {path} is too large ({size} bytes, max {max})")]
    TooLarge {
        path: PathBuf,
        size: u64,
        max: usize,
    },
    #[error("Failed to read context file {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Context file {path} is empty")]
    Empty { path: PathBuf },
}

/// Result of loading project context
#[derive(Debug, Clone)]
pub struct ProjectContext {
    /// The loaded instructions content
    pub instructions: Option<String>,
    /// Path to the loaded file (for display)
    pub source_path: Option<PathBuf>,
    /// Any warnings during loading
    pub warnings: Vec<String>,
    /// Project root directory
    #[allow(dead_code)] // Part of ProjectContext public interface
    pub project_root: PathBuf,
    /// Whether this is a trusted project
    pub is_trusted: bool,
}

impl ProjectContext {
    /// Create an empty project context
    pub fn empty(project_root: PathBuf) -> Self {
        Self {
            instructions: None,
            source_path: None,
            warnings: Vec::new(),
            project_root,
            is_trusted: false,
        }
    }

    /// Check if any instructions were loaded
    pub fn has_instructions(&self) -> bool {
        self.instructions.is_some()
    }

    /// Get the instructions as a formatted block for system prompt
    pub fn as_system_block(&self) -> Option<String> {
        self.instructions.as_ref().map(|content| {
            let source = self
                .source_path
                .as_ref()
                .map_or_else(|| "project".to_string(), |p| p.display().to_string());

            format!(
                "<project_instructions source=\"{source}\">\n{content}\n</project_instructions>"
            )
        })
    }
}

#[derive(Debug, Serialize)]
struct ProjectContextPack {
    project_name: String,
    directory_structure: Vec<String>,
    readme: Option<ReadmePack>,
    config_files: Vec<String>,
    key_source_files: Vec<String>,
    counts: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize)]
struct ReadmePack {
    path: String,
    excerpt: String,
}

/// Generate a deterministic, cache-friendly project context pack.
///
/// The pack intentionally uses only stable workspace facts: relative paths,
/// sorted entries, bounded README text, and sorted JSON object fields. It does
/// not include timestamps, random ids, absolute temp paths, or live git state.
pub fn generate_project_context_pack(workspace: &Path) -> Option<String> {
    let cache_key = fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());

    // Always walk the directory to compute entries + readme excerpt.
    // The manifest_hash is the authoritative cache key — it covers
    // the file list AND the README excerpt content, so it catches
    // content changes that directory mtime alone would miss.
    let mut entries = Vec::new();
    collect_pack_entries(workspace, &mut entries);
    sort_pack_paths(&mut entries);
    entries.truncate(PACK_MAX_ENTRIES);

    let readme = read_readme_excerpt(workspace, &entries);
    let manifest_hash = project_pack_manifest_hash(&entries, readme.as_ref());

    // Check cache: if manifest_hash matches, reuse the rendered pack.
    if let Some(cached) = PROJECT_PACK_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()
        .and_then(|cache| cache.get(&cache_key).cloned())
        && cached.manifest_hash == manifest_hash
    {
        return Some(cached.rendered);
    }

    let mut config_files = entries
        .iter()
        .filter(|path| is_config_file(path))
        .take(PACK_MAX_CONFIG_FILES)
        .cloned()
        .collect::<Vec<_>>();
    sort_pack_paths(&mut config_files);

    let mut key_source_files = entries
        .iter()
        .filter(|path| is_source_file(path))
        .take(PACK_MAX_SOURCE_FILES)
        .cloned()
        .collect::<Vec<_>>();
    sort_pack_paths(&mut key_source_files);

    let mut counts = BTreeMap::new();
    counts.insert("config_files".to_string(), config_files.len());
    counts.insert("directory_entries".to_string(), entries.len());
    counts.insert("key_source_files".to_string(), key_source_files.len());

    let pack = ProjectContextPack {
        project_name: workspace
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("workspace")
            .to_string(),
        directory_structure: entries,
        readme,
        config_files,
        key_source_files,
        counts,
    };

    let json = serde_json::to_string_pretty(&pack).ok()?;
    let rendered = format!(
        "## Project Context Pack\n\n<project_context_pack>\n{json}\n</project_context_pack>"
    );
    if let Ok(mut cache) = PROJECT_PACK_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        cache.insert(
            cache_key,
            CachedProjectPack {
                manifest_hash,
                rendered: rendered.clone(),
            },
        );
    }
    Some(rendered)
}

fn collect_pack_entries(root: &Path, out: &mut Vec<String>) {
    let mut builder = WalkBuilder::new(root);
    let root_for_filter = root.to_path_buf();
    builder
        .max_depth(Some(PACK_MAX_DEPTH + 1))
        .follow_links(false)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(false)
        .require_git(false)
        .filter_entry(move |entry| should_walk_pack_entry(&root_for_filter, entry));
    let _ = builder.add_custom_ignore_filename(".deepseekignore");

    for result in builder.build() {
        let Ok(entry) = result else {
            continue;
        };
        if entry.depth() == 0 {
            continue;
        }
        let Some(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue;
        }
        let Some(relative) = relative_slash_path(root, entry.path()) else {
            continue;
        };
        if is_ignored_pack_path(&relative, file_type.is_dir()) {
            continue;
        }
        if file_type.is_dir() {
            out.push(format!("{relative}/"));
        } else if file_type.is_file() {
            out.push(relative);
        }
    }
}

fn should_walk_pack_entry(root: &Path, entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }
    let Some(file_type) = entry.file_type() else {
        return false;
    };
    if file_type.is_symlink() {
        return false;
    }
    let Some(relative) = relative_slash_path(root, entry.path()) else {
        return false;
    };
    !is_ignored_pack_path(&relative, file_type.is_dir())
}

fn relative_slash_path(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    let mut parts = Vec::new();
    for component in relative.components() {
        parts.push(component.as_os_str().to_string_lossy().to_string());
    }
    normalize_pack_relative_path(&parts.join("/"))
}

fn normalize_pack_relative_path(path: &str) -> Option<String> {
    let normalized = path.replace('\\', "/");
    let mut parts = Vec::new();
    for part in normalized.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return None;
        }
        parts.push(part);
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

fn sort_pack_paths(paths: &mut [String]) {
    paths.sort_by(|a, b| {
        pack_path_priority(a)
            .cmp(&pack_path_priority(b))
            .then_with(|| pack_path_sort_key(a).cmp(&pack_path_sort_key(b)))
            .then_with(|| a.cmp(b))
    });
}

fn pack_path_sort_key(path: &str) -> String {
    path.replace('\\', "/").to_ascii_lowercase()
}

fn pack_path_priority(path: &str) -> u8 {
    let lower = pack_path_sort_key(path);
    let name = lower.trim_end_matches('/').rsplit('/').next().unwrap_or("");
    if matches!(name, "readme.md" | "readme.txt" | "readme") {
        0
    } else if is_config_file(&lower) {
        1
    } else if is_source_file(&lower) {
        2
    } else if lower.ends_with('/') {
        3
    } else {
        4
    }
}

fn is_ignored_pack_path(relative: &str, is_dir: bool) -> bool {
    let normalized = relative.trim_end_matches('/');
    let lower = normalized.to_ascii_lowercase();
    if lower
        .split('/')
        .any(|part| PACK_IGNORED_DIRS.contains(&part))
    {
        return true;
    }
    if is_dir {
        return false;
    }
    let name = lower.rsplit('/').next().unwrap_or(lower.as_str());
    PACK_IGNORED_FILES.contains(&name)
        || name.ends_with(".log")
        || name.ends_with(".tmp")
        || name.ends_with(".temp")
        || name.ends_with(".swp")
        || name.ends_with(".swo")
        || name.ends_with(".bak")
        || name.ends_with('~')
        || name.starts_with(".#")
}

fn project_pack_manifest_hash(entries: &[String], readme: Option<&ReadmePack>) -> String {
    let mut hasher = Sha256::new();
    for entry in entries {
        hasher.update(entry.as_bytes());
        hasher.update([0]);
    }
    if let Some(readme) = readme {
        hasher.update(readme.path.as_bytes());
        hasher.update([0]);
        hasher.update(readme.excerpt.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn read_readme_excerpt(workspace: &Path, entries: &[String]) -> Option<ReadmePack> {
    let path = entries
        .iter()
        .find(|path| {
            let lower = path.to_ascii_lowercase();
            lower == "readme.md" || lower == "readme.txt" || lower == "readme"
        })?
        .clone();
    let raw = fs::read_to_string(workspace.join(&path)).ok()?;
    let excerpt = truncate_chars(raw.trim(), PACK_README_MAX_CHARS);
    if excerpt.is_empty() {
        None
    } else {
        Some(ReadmePack { path, excerpt })
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    value.chars().take(max_chars).collect::<String>()
}

fn is_config_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(lower.as_str());
    matches!(
        name,
        "cargo.toml"
            | "package.json"
            | "tsconfig.json"
            | "pyproject.toml"
            | "requirements.txt"
            | "go.mod"
            | "config.toml"
            | "deepseek.toml"
            | "dockerfile"
            | "compose.yaml"
            | "compose.yml"
            | "docker-compose.yaml"
            | "docker-compose.yml"
            | "makefile"
    ) || lower.ends_with(".config.js")
        || lower.ends_with(".config.ts")
        || lower.ends_with(".toml")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
}

fn is_source_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    matches!(
        lower.rsplit('.').next(),
        Some(
            "rs" | "py"
                | "js"
                | "jsx"
                | "ts"
                | "tsx"
                | "go"
                | "java"
                | "kt"
                | "c"
                | "cc"
                | "cpp"
                | "h"
                | "hpp"
                | "cs"
                | "rb"
                | "php"
                | "swift"
                | "sql"
                | "sh"
                | "bash"
        )
    )
}

/// Load project context from the workspace directory.
///
/// This searches for known project context files and loads the first one found.
pub fn load_project_context(workspace: &Path) -> ProjectContext {
    let mut ctx = ProjectContext::empty(workspace.to_path_buf());

    // Search for project context files
    for filename in PROJECT_CONTEXT_FILES {
        let file_path = workspace.join(filename);

        if file_path.exists() && file_path.is_file() {
            match load_context_file(&file_path) {
                Ok(content) => {
                    ctx.instructions = Some(content);
                    ctx.source_path = Some(file_path);
                    break;
                }
                Err(error) => {
                    ctx.warnings.push(error.to_string());
                }
            }
        }
    }

    // Check for trust file
    ctx.is_trusted = check_trust_status(workspace);

    ctx
}

/// Load project context from parent directories as well.
///
/// This allows for monorepo setups where a root AGENTS.md applies to all subdirectories.
pub fn load_project_context_with_parents(workspace: &Path) -> ProjectContext {
    load_project_context_with_parents_and_home(workspace, dirs::home_dir().as_deref())
}

fn load_project_context_with_parents_and_home(
    workspace: &Path,
    home_dir: Option<&Path>,
) -> ProjectContext {
    let mut ctx = load_project_context(workspace);

    // If no context found in workspace, check parent directories
    if !ctx.has_instructions() {
        let mut current = workspace.parent();

        while let Some(parent) = current {
            let parent_ctx = load_project_context(parent);
            ctx.warnings.extend(parent_ctx.warnings.iter().cloned());
            if parent_ctx.has_instructions() {
                ctx.instructions = parent_ctx.instructions;
                ctx.source_path = parent_ctx.source_path;
                break;
            }

            current = parent.parent();
        }
    }

    if !ctx.has_instructions()
        && let Some(global_ctx) = load_global_agents_context(workspace, home_dir)
    {
        ctx.warnings.extend(global_ctx.warnings.iter().cloned());
        if global_ctx.has_instructions() {
            ctx.instructions = global_ctx.instructions;
            ctx.source_path = global_ctx.source_path;
        }
    }

    // Auto-generate .deepseek/instructions.md when no context file exists anywhere.
    // This avoids the per-turn filesystem scan fallback in prompts.rs that
    // breaks KV prefix cache stability.
    if !ctx.has_instructions()
        && let Some(generated) = auto_generate_context(workspace)
    {
        let mut warnings = std::mem::take(&mut ctx.warnings);
        ctx = load_project_context(workspace);
        warnings.extend(ctx.warnings.iter().cloned());
        ctx.warnings = warnings;
        if !ctx.has_instructions() {
            // Loaded from the file we just wrote — use the generated content
            // directly as a last resort (shouldn't normally happen).
            ctx.instructions = Some(generated);
            ctx.source_path = None;
        }
    }

    ctx
}

fn load_global_agents_context(workspace: &Path, home_dir: Option<&Path>) -> Option<ProjectContext> {
    let home = home_dir?;
    let mut path = home.to_path_buf();
    for component in GLOBAL_AGENTS_RELATIVE_PATH {
        path.push(component);
    }

    if !(path.exists() && path.is_file()) {
        return None;
    }

    let mut ctx = ProjectContext::empty(workspace.to_path_buf());
    match load_context_file(&path) {
        Ok(content) => {
            ctx.instructions = Some(content);
            ctx.source_path = Some(path);
        }
        Err(error) => ctx.warnings.push(error.to_string()),
    }
    Some(ctx)
}

/// Generate a context file from project tree + summary and write it to
/// `.deepseek/instructions.md`. Returns the generated content on success.
fn auto_generate_context(workspace: &Path) -> Option<String> {
    let deepseek_dir = workspace.join(".deepseek");
    let instructions_path = deepseek_dir.join("instructions.md");

    // Don't overwrite an existing file
    if instructions_path.exists() {
        return None;
    }

    let summary = crate::utils::summarize_project(workspace);
    let tree = crate::utils::project_tree(workspace, 2);

    let content = format!(
        "# Project Structure (Auto-generated)\n\n\
         > This file was automatically generated by DeepSeek TUI.\n\
         > You can edit or delete it at any time.\n\n\
         **Summary:** {summary}\n\n\
         **Tree:**\n```\n{tree}\n```"
    );

    // Create .deepseek/ directory if needed
    if let Err(e) = std::fs::create_dir_all(&deepseek_dir) {
        tracing::warn!("Failed to create .deepseek/ directory: {e}");
        return None;
    }

    match std::fs::write(&instructions_path, &content) {
        Ok(()) => {
            tracing::info!("Auto-generated {}", instructions_path.display());
            Some(content)
        }
        Err(e) => {
            tracing::warn!("Failed to write {}: {e}", instructions_path.display());
            None
        }
    }
}

/// Load a context file with size checking
fn load_context_file(path: &Path) -> Result<String, ProjectContextError> {
    // Check file size first
    let metadata = fs::metadata(path).map_err(|source| ProjectContextError::Metadata {
        path: path.to_path_buf(),
        source,
    })?;

    if metadata.len() > MAX_CONTEXT_SIZE as u64 {
        return Err(ProjectContextError::TooLarge {
            path: path.to_path_buf(),
            size: metadata.len(),
            max: MAX_CONTEXT_SIZE,
        });
    }

    // Read the file
    let content = fs::read_to_string(path).map_err(|source| ProjectContextError::Read {
        path: path.to_path_buf(),
        source,
    })?;

    // Basic validation
    if content.trim().is_empty() {
        return Err(ProjectContextError::Empty {
            path: path.to_path_buf(),
        });
    }

    Ok(content)
}

/// Check if this project is marked as trusted
fn check_trust_status(workspace: &Path) -> bool {
    if crate::config::is_workspace_trusted(workspace) {
        return true;
    }

    // Check for trust markers
    let trust_markers = [
        workspace.join(".deepseek").join("trusted"),
        workspace.join(".deepseek").join("trust.json"),
    ];

    for marker in &trust_markers {
        if marker.exists() {
            return true;
        }
    }

    false
}

/// Create a default AGENTS.md file for a project
pub fn create_default_agents_md(workspace: &Path) -> std::io::Result<PathBuf> {
    let agents_path = workspace.join("AGENTS.md");

    let default_content = r#"# Project Agent Instructions

This file provides guidance to AI agents (DeepSeek TUI, Claude Code, etc.) when working with code in this repository.

## File Location

Save this file as `AGENTS.md` in your project root so the CLI can load it automatically.

## Build and Development Commands

```bash
# Build
# cargo build              # Rust projects
# npm run build            # Node.js projects
# python -m build          # Python projects

# Test
# cargo test               # Rust
# npm test                 # Node.js
# pytest                   # Python

# Lint and Format
# cargo fmt && cargo clippy  # Rust
# npm run lint               # Node.js
# ruff check .               # Python
```

## Architecture Overview

<!-- Describe your project's high-level architecture here -->
<!-- Focus on the "big picture" that requires reading multiple files to understand -->

### Key Components

<!-- List and describe the main components/modules -->

### Data Flow

<!-- Describe how data flows through the system -->

## Configuration Files

<!-- List important configuration files and their purposes -->

## Extension Points

<!-- Describe how to extend the codebase (add new features, tools, etc.) -->

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
"#;

    fs::write(&agents_path, default_content)?;
    Ok(agents_path)
}

/// Merge multiple project contexts (e.g., from nested directories)
#[allow(dead_code)] // Public API for monorepo context merging
pub fn merge_contexts(contexts: &[ProjectContext]) -> Option<String> {
    let non_empty: Vec<_> = contexts
        .iter()
        .filter_map(ProjectContext::as_system_block)
        .collect();

    if non_empty.is_empty() {
        None
    } else {
        Some(non_empty.join("\n\n"))
    }
}

// === Unit Tests ===

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use tempfile::tempdir;

    fn sha256_hex(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn project_pack_hash(workspace: &Path) -> String {
        sha256_hex(&generate_project_context_pack(workspace).expect("pack"))
    }

    #[test]
    fn test_load_project_context_empty() {
        let tmp = tempdir().expect("tempdir");
        let ctx = load_project_context(tmp.path());

        assert!(!ctx.has_instructions());
        assert!(ctx.source_path.is_none());
    }

    #[test]
    fn test_load_project_context_agents_md() {
        let tmp = tempdir().expect("tempdir");
        let agents_path = tmp.path().join("AGENTS.md");
        fs::write(&agents_path, "# Test Instructions\n\nFollow these rules.").expect("write");

        let ctx = load_project_context(tmp.path());

        assert!(ctx.has_instructions());
        assert!(
            ctx.instructions
                .as_ref()
                .unwrap()
                .contains("Test Instructions")
        );
        assert_eq!(ctx.source_path, Some(agents_path));
    }

    #[test]
    fn test_load_project_context_priority() {
        let tmp = tempdir().expect("tempdir");

        // Create both files - AGENTS.md should take priority
        fs::write(tmp.path().join("AGENTS.md"), "AGENTS content").expect("write");
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir(&claude_dir).expect("mkdir");
        fs::write(claude_dir.join("instructions.md"), "CLAUDE content").expect("write");

        let ctx = load_project_context(tmp.path());

        assert!(ctx.has_instructions());
        assert!(
            ctx.instructions
                .as_ref()
                .unwrap()
                .contains("AGENTS content")
        );
    }

    #[test]
    fn test_load_project_context_hidden_dir() {
        let tmp = tempdir().expect("tempdir");
        let hidden_dir = tmp.path().join(".deepseek");
        fs::create_dir(&hidden_dir).expect("mkdir");
        fs::write(hidden_dir.join("instructions.md"), "Hidden instructions").expect("write");

        let ctx = load_project_context(tmp.path());

        assert!(ctx.has_instructions());
        assert!(
            ctx.instructions
                .as_ref()
                .unwrap()
                .contains("Hidden instructions")
        );
    }

    #[test]
    fn test_as_system_block() {
        let tmp = tempdir().expect("tempdir");
        let agents_path = tmp.path().join("AGENTS.md");
        fs::write(&agents_path, "Test content").expect("write");

        let ctx = load_project_context(tmp.path());
        let block = ctx.as_system_block().expect("block");

        assert!(block.contains("<project_instructions"));
        assert!(block.contains("Test content"));
        assert!(block.contains("</project_instructions>"));
    }

    #[test]
    fn test_empty_file_warning() {
        let tmp = tempdir().expect("tempdir");
        let agents_path = tmp.path().join("AGENTS.md");
        fs::write(&agents_path, "   \n  \n  ").expect("write"); // Only whitespace

        let ctx = load_project_context(tmp.path());

        assert!(!ctx.has_instructions());
        assert!(!ctx.warnings.is_empty());
    }

    #[test]
    fn test_check_trust_status() {
        let tmp = tempdir().expect("tempdir");

        // Not trusted by default
        assert!(!check_trust_status(tmp.path()));

        // Create trust marker
        let deepseek_dir = tmp.path().join(".deepseek");
        fs::create_dir(&deepseek_dir).expect("mkdir");
        fs::write(deepseek_dir.join("trusted"), "").expect("write");

        assert!(check_trust_status(tmp.path()));
    }

    #[test]
    fn test_create_default_agents_md() {
        let tmp = tempdir().expect("tempdir");
        let path = create_default_agents_md(tmp.path()).expect("create");

        assert!(path.exists());
        let content = fs::read_to_string(&path).expect("read");
        assert!(content.contains("Project Agent Instructions"));
    }

    #[test]
    fn test_load_with_parents() {
        let tmp = tempdir().expect("tempdir");

        // Create a nested structure
        let subdir = tmp.path().join("subproject");
        fs::create_dir(&subdir).expect("mkdir");

        // Put AGENTS.md in parent
        fs::write(tmp.path().join("AGENTS.md"), "Parent instructions").expect("write");
        // Also create .git to mark as repo root
        fs::create_dir(tmp.path().join(".git")).expect("mkdir .git");

        // Load from subdir should find parent's AGENTS.md
        let ctx = load_project_context_with_parents(&subdir);

        assert!(ctx.has_instructions());
        assert!(
            ctx.instructions
                .as_ref()
                .unwrap()
                .contains("Parent instructions")
        );
    }

    #[test]
    fn test_merge_contexts() {
        let mut ctx1 = ProjectContext::empty(PathBuf::from("/a"));
        ctx1.instructions = Some("Instructions A".to_string());
        ctx1.source_path = Some(PathBuf::from("/a/AGENTS.md"));

        let mut ctx2 = ProjectContext::empty(PathBuf::from("/b"));
        ctx2.instructions = Some("Instructions B".to_string());
        ctx2.source_path = Some(PathBuf::from("/b/AGENTS.md"));

        let merged = merge_contexts(&[ctx1, ctx2]).expect("merge");

        assert!(merged.contains("Instructions A"));
        assert!(merged.contains("Instructions B"));
    }

    #[test]
    fn test_load_with_parents_searches_above_git_root_when_needed() {
        let tmp = tempdir().expect("tempdir");

        // AGENTS.md exists above repository root.
        fs::write(tmp.path().join("AGENTS.md"), "Organization instructions").expect("write");

        // Mark repository root one level below.
        let repo_root = tmp.path().join("repo");
        fs::create_dir(&repo_root).expect("mkdir repo");
        fs::create_dir(repo_root.join(".git")).expect("mkdir .git");

        let workspace = repo_root.join("apps").join("client");
        fs::create_dir_all(&workspace).expect("mkdir workspace");

        let ctx = load_project_context_with_parents(&workspace);
        assert!(ctx.has_instructions());
        assert!(
            ctx.instructions
                .as_ref()
                .unwrap()
                .contains("Organization instructions")
        );
    }

    #[test]
    fn project_context_pack_is_stable_and_sorted() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Demo\n\nReadme body").expect("write");
        fs::write(tmp.path().join("Cargo.toml"), "[package]\nname = \"demo\"").expect("write");
        fs::create_dir_all(tmp.path().join("src")).expect("mkdir src");
        fs::write(tmp.path().join("src").join("z.rs"), "mod z;").expect("write z");
        fs::write(tmp.path().join("src").join("a.rs"), "mod a;").expect("write a");
        fs::create_dir_all(tmp.path().join("node_modules").join("pkg")).expect("mkdir ignored");
        fs::write(
            tmp.path().join("node_modules").join("pkg").join("index.js"),
            "ignored",
        )
        .expect("write ignored");

        let first = generate_project_context_pack(tmp.path()).expect("pack");
        let second = generate_project_context_pack(tmp.path()).expect("pack again");

        assert_eq!(first, second);
        assert_eq!(
            sha256_hex(&first),
            sha256_hex(&second),
            "same project context must produce the same project pack hash"
        );
        assert!(first.contains("\"project_name\""));
        assert!(first.contains("\"directory_structure\""));
        assert!(first.contains("\"README.md\""));
        assert!(first.contains("\"Cargo.toml\""));
        assert!(first.contains("\"src/a.rs\""));
        assert!(first.contains("\"src/z.rs\""));
        assert!(!first.contains("node_modules"));
        assert!(
            first.find("\"src/a.rs\"").expect("a before z")
                < first.find("\"src/z.rs\"").expect("z")
        );
    }

    #[test]
    fn project_context_pack_is_stable_across_creation_order() {
        let left_root = tempdir().expect("left tempdir");
        let right_root = tempdir().expect("right tempdir");
        let left = left_root.path().join("repo");
        let right = right_root.path().join("repo");

        fs::create_dir_all(left.join("src")).expect("mkdir left src");
        fs::write(left.join("README.md"), "# Demo\n\nStable README").expect("left readme");
        fs::write(left.join("Cargo.toml"), "[package]\nname = \"demo\"").expect("left cargo");
        fs::write(left.join("src").join("z.rs"), "mod z;").expect("left z");
        fs::write(left.join("src").join("a.rs"), "mod a;").expect("left a");

        fs::create_dir_all(right.join("src")).expect("mkdir right src");
        fs::write(right.join("src").join("a.rs"), "mod a;").expect("right a");
        fs::write(right.join("src").join("z.rs"), "mod z;").expect("right z");
        fs::write(right.join("Cargo.toml"), "[package]\nname = \"demo\"").expect("right cargo");
        fs::write(right.join("README.md"), "# Demo\n\nStable README").expect("right readme");

        assert_eq!(
            generate_project_context_pack(&left),
            generate_project_context_pack(&right)
        );
    }

    #[test]
    fn project_context_pack_ignores_volatile_paths() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Demo\n\nReadme body").expect("write readme");
        fs::create_dir_all(tmp.path().join("src")).expect("mkdir src");
        fs::write(tmp.path().join("src").join("lib.rs"), "pub fn stable() {}").expect("write lib");

        let before = project_pack_hash(tmp.path());

        for dir in [
            ".git/objects",
            "target/debug",
            "node_modules/pkg",
            "dist/assets",
            "build/output",
            ".next/cache",
            "__pycache__",
        ] {
            fs::create_dir_all(tmp.path().join(dir)).expect("mkdir ignored dir");
        }
        fs::write(tmp.path().join(".git").join("HEAD"), "ref: refs/heads/main").expect("write git");
        fs::write(
            tmp.path().join("target").join("debug").join("build.log"),
            "log",
        )
        .expect("write target");
        fs::write(
            tmp.path().join("node_modules").join("pkg").join("index.js"),
            "ignored",
        )
        .expect("write node_modules");
        fs::write(
            tmp.path().join("dist").join("assets").join("app.js"),
            "dist",
        )
        .expect("write dist");
        fs::write(
            tmp.path().join("build").join("output").join("app.js"),
            "build",
        )
        .expect("write build");
        fs::write(tmp.path().join(".next").join("cache").join("page"), "next").expect("write next");
        fs::write(tmp.path().join("__pycache__").join("mod.pyc"), "pyc").expect("write pycache");
        fs::write(tmp.path().join("run.log"), "log").expect("write log");
        fs::write(tmp.path().join("scratch.tmp"), "tmp").expect("write tmp");

        let after_pack = generate_project_context_pack(tmp.path()).expect("pack after ignores");
        assert_eq!(before, sha256_hex(&after_pack));
        for ignored in [
            ".git",
            "target",
            "node_modules",
            "dist",
            "build",
            ".next",
            "__pycache__",
            "run.log",
            "scratch.tmp",
        ] {
            assert!(
                !after_pack.contains(ignored),
                "pack should not include ignored path {ignored}"
            );
        }
    }

    #[test]
    fn project_context_pack_readme_hash_is_stable_when_content_is_unchanged() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Demo\n\nStable body").expect("write readme");
        fs::create_dir_all(tmp.path().join("src")).expect("mkdir src");
        fs::write(tmp.path().join("src").join("lib.rs"), "pub fn stable() {}").expect("write lib");

        let before = project_pack_hash(tmp.path());
        fs::write(tmp.path().join("README.md"), "# Demo\n\nStable body").expect("rewrite readme");
        let after = project_pack_hash(tmp.path());

        assert_eq!(before, after);
    }

    #[test]
    fn project_context_pack_normalizes_windows_and_unix_paths_for_sorting() {
        let mut windows_paths = vec![
            normalize_pack_relative_path(r"src\z.rs").expect("normalize z"),
            normalize_pack_relative_path(r".\src\a.rs").expect("normalize a"),
            normalize_pack_relative_path(r"config\DeepSeek.toml").expect("normalize config"),
        ];
        let mut unix_paths = vec![
            normalize_pack_relative_path("src/z.rs").expect("normalize z"),
            normalize_pack_relative_path("./src/a.rs").expect("normalize a"),
            normalize_pack_relative_path("config/DeepSeek.toml").expect("normalize config"),
        ];

        sort_pack_paths(&mut windows_paths);
        sort_pack_paths(&mut unix_paths);

        assert_eq!(windows_paths, unix_paths);
        assert_eq!(
            unix_paths,
            vec![
                "config/DeepSeek.toml".to_string(),
                "src/a.rs".to_string(),
                "src/z.rs".to_string(),
            ]
        );
    }

    #[test]
    fn project_context_pack_respects_gitignore_and_deepseekignore() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Demo").expect("write readme");
        fs::write(tmp.path().join(".gitignore"), "ignored-by-git.rs\n").expect("write gitignore");
        fs::write(
            tmp.path().join(".deepseekignore"),
            "ignored-by-deepseek.rs\n",
        )
        .expect("write deepseekignore");
        fs::write(tmp.path().join("ignored-by-git.rs"), "ignored").expect("write git ignored");
        fs::write(tmp.path().join("ignored-by-deepseek.rs"), "ignored")
            .expect("write deepseek ignored");
        fs::write(tmp.path().join("included.rs"), "included").expect("write included");

        let pack = generate_project_context_pack(tmp.path()).expect("pack");

        assert!(pack.contains("included.rs"));
        assert!(!pack.contains("ignored-by-git.rs"));
        assert!(!pack.contains("ignored-by-deepseek.rs"));
    }

    #[test]
    fn project_context_pack_does_not_include_symlink_outside_workspace() {
        let tmp = tempdir().expect("workspace tempdir");
        let outside = tempdir().expect("outside tempdir");
        fs::write(tmp.path().join("README.md"), "# Demo").expect("write readme");
        fs::write(
            outside.path().join("secret.rs"),
            "pub const SECRET: &str = \"outside\";",
        )
        .expect("write outside");

        let link_path = tmp.path().join("linked-secret.rs");
        if !try_symlink_file(&outside.path().join("secret.rs"), &link_path) {
            return;
        }

        let pack = generate_project_context_pack(tmp.path()).expect("pack");
        assert!(!pack.contains("linked-secret.rs"));
        assert!(!pack.contains("SECRET"));
    }

    #[cfg(unix)]
    fn try_symlink_file(target: &Path, link: &Path) -> bool {
        std::os::unix::fs::symlink(target, link).is_ok()
    }

    #[cfg(windows)]
    fn try_symlink_file(target: &Path, link: &Path) -> bool {
        std::os::windows::fs::symlink_file(target, link).is_ok()
    }

    #[test]
    fn test_load_global_agents_when_project_has_no_context() {
        let workspace = tempdir().expect("workspace tempdir");
        let home = tempdir().expect("home tempdir");
        let global_dir = home.path().join(".deepseek");
        fs::create_dir(&global_dir).expect("mkdir .deepseek");
        let global_agents = global_dir.join("AGENTS.md");
        fs::write(&global_agents, "Global instructions").expect("write global agents");

        let ctx = load_project_context_with_parents_and_home(workspace.path(), Some(home.path()));

        assert!(ctx.has_instructions());
        assert!(
            ctx.instructions
                .as_ref()
                .unwrap()
                .contains("Global instructions")
        );
        assert_eq!(ctx.source_path, Some(global_agents));
    }

    #[test]
    fn test_local_agents_takes_priority_over_global_agents() {
        let workspace = tempdir().expect("workspace tempdir");
        fs::write(workspace.path().join("AGENTS.md"), "Local instructions")
            .expect("write local agents");

        let home = tempdir().expect("home tempdir");
        let global_dir = home.path().join(".deepseek");
        fs::create_dir(&global_dir).expect("mkdir .deepseek");
        fs::write(global_dir.join("AGENTS.md"), "Global instructions")
            .expect("write global agents");

        let ctx = load_project_context_with_parents_and_home(workspace.path(), Some(home.path()));

        assert!(ctx.has_instructions());
        let instructions = ctx.instructions.as_ref().unwrap();
        assert!(instructions.contains("Local instructions"));
        assert!(!instructions.contains("Global instructions"));
        assert_eq!(ctx.source_path, Some(workspace.path().join("AGENTS.md")));
    }

    #[test]
    fn test_invalid_global_agents_warns_and_falls_back_to_generated_context() {
        let workspace = tempdir().expect("workspace tempdir");
        let home = tempdir().expect("home tempdir");
        let global_dir = home.path().join(".deepseek");
        fs::create_dir(&global_dir).expect("mkdir .deepseek");
        fs::write(global_dir.join("AGENTS.md"), "   \n  ").expect("write empty global agents");

        let ctx = load_project_context_with_parents_and_home(workspace.path(), Some(home.path()));

        assert!(
            ctx.warnings
                .iter()
                .any(|warning| warning.contains("Context file") && warning.contains("is empty")),
            "expected empty global AGENTS.md warning, got {:?}",
            ctx.warnings
        );
        assert!(ctx.has_instructions());
        assert!(
            ctx.instructions
                .as_ref()
                .unwrap()
                .contains("Project Structure (Auto-generated)")
        );
    }

    #[test]
    fn project_context_pack_hash_changes_when_readme_content_changes() {
        let tmp = tempdir().expect("tempdir");
        fs::write(
            tmp.path().join("README.md"),
            "# Original README\n\nOriginal body.",
        )
        .expect("write readme");
        fs::create_dir_all(tmp.path().join("src")).expect("mkdir src");
        fs::write(tmp.path().join("src").join("lib.rs"), "pub fn stable() {}").expect("write lib");

        let before = project_pack_hash(tmp.path());

        // Change README content — hash must change.
        fs::write(
            tmp.path().join("README.md"),
            "# Modified README\n\nNew body content.",
        )
        .expect("rewrite readme");
        let after = project_pack_hash(tmp.path());

        assert_ne!(
            before, after,
            "project_pack_hash must change when README content changes"
        );
    }

    #[test]
    fn project_context_pack_hash_stable_when_readme_unchanged() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Stable README").expect("write readme");
        fs::create_dir_all(tmp.path().join("src")).expect("mkdir src");
        fs::write(tmp.path().join("src").join("main.rs"), "fn main() {}").expect("write main");

        let first = project_pack_hash(tmp.path());
        // Touch a non-readme file without changing its content.
        let second = project_pack_hash(tmp.path());

        assert_eq!(first, second, "hash must be stable when nothing changes");
    }

    #[test]
    fn project_context_pack_hash_ignores_target_directory_changes() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Demo").expect("write readme");
        fs::create_dir_all(tmp.path().join("src")).expect("mkdir src");
        fs::write(tmp.path().join("src").join("lib.rs"), "pub fn x() {}").expect("write lib");

        let before = project_pack_hash(tmp.path());

        // Add files to target/ (ignored directory).
        fs::create_dir_all(tmp.path().join("target").join("debug")).expect("mkdir target");
        fs::write(
            tmp.path().join("target").join("debug").join("binary"),
            "binary content",
        )
        .expect("write target file");

        let after = project_pack_hash(tmp.path());
        assert_eq!(
            before, after,
            "hash must not change when ignored directory (target/) changes"
        );
    }

    #[test]
    fn project_context_pack_hash_changes_when_new_file_added() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Demo").expect("write readme");
        fs::create_dir_all(tmp.path().join("src")).expect("mkdir src");
        fs::write(tmp.path().join("src").join("lib.rs"), "pub fn x() {}").expect("write lib");

        let before = project_pack_hash(tmp.path());

        // Add a new source file that will appear in the pack.
        fs::write(
            tmp.path().join("src").join("new_module.rs"),
            "pub fn new() {}",
        )
        .expect("write new module");

        let after = project_pack_hash(tmp.path());
        assert_ne!(
            before, after,
            "hash must change when a new file is added to the pack"
        );
    }

    #[test]
    fn project_context_pack_hash_changes_when_readme_excerpt_changes() {
        let tmp = tempdir().expect("tempdir");
        let long_readme = "A".repeat(5000);
        fs::write(tmp.path().join("README.md"), &long_readme).expect("write long readme");

        let before = project_pack_hash(tmp.path());

        // Change only the first 4000 chars (which become the excerpt).
        let mut new_readme = "B".repeat(4000);
        new_readme.push_str(&"A".repeat(1000));
        fs::write(tmp.path().join("README.md"), &new_readme).expect("write modified readme");

        let after = project_pack_hash(tmp.path());
        assert_ne!(
            before, after,
            "hash must change when README excerpt content changes"
        );
    }

    #[test]
    fn project_context_pack_hash_ignored_file_changes() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Demo").expect("write readme");
        fs::create_dir_all(tmp.path().join("src")).expect("mkdir src");
        fs::write(tmp.path().join("src").join("lib.rs"), "pub fn x() {}").expect("write lib");

        let before = project_pack_hash(tmp.path());

        // Add a .DS_Store file (ignored by PACK_IGNORED_FILES).
        fs::write(tmp.path().join(".DS_Store"), "store data").expect("write ds_store");

        let after = project_pack_hash(tmp.path());
        assert_eq!(
            before, after,
            "hash must not change when an ignored file (.DS_Store) is added"
        );
    }

    #[test]
    fn project_context_pack_truncation_is_deterministic() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Big project").expect("write readme");

        // Create more files than PACK_MAX_ENTRIES (400).
        for i in 0..500 {
            let name = format!("file_{:04}.rs", i);
            fs::write(tmp.path().join(&name), format!("// file {i}")).expect("write file");
        }

        // Run twice — results must be identical (sort-then-truncate is stable).
        let first = project_pack_hash(tmp.path());
        let second = project_pack_hash(tmp.path());
        assert_eq!(
            first, second,
            "pack hash must be deterministic with >400 candidates"
        );
    }

    #[test]
    fn project_context_pack_config_files_not_squeezed_out() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join("README.md"), "# Project").expect("write readme");

        // Create many ordinary source files that sort before config files.
        for i in 0..100 {
            let name = format!("aaa_{:03}.rs", i);
            fs::write(tmp.path().join(&name), format!("// {i}")).expect("write file");
        }

        // Create a recognized config file (Cargo.toml).
        fs::write(tmp.path().join("Cargo.toml"), "[package]\nname = \"demo\"")
            .expect("write cargo");

        let pack = generate_project_context_pack(tmp.path()).expect("pack");
        // Cargo.toml is a recognized config file and must appear in the pack.
        assert!(
            pack.contains("Cargo.toml"),
            "config file Cargo.toml must be present in pack"
        );
    }

    #[test]
    fn project_context_pack_prioritizes_readme_and_config_before_truncation() {
        let tmp = tempdir().expect("tempdir");

        for i in 0..600 {
            let name = format!("aaa_{:03}.rs", i);
            fs::write(tmp.path().join(&name), format!("// {i}")).expect("write file");
        }
        fs::write(tmp.path().join("README.md"), "# Project").expect("write readme");
        fs::write(tmp.path().join("Cargo.toml"), "[package]\nname = \"demo\"")
            .expect("write cargo");

        let pack = generate_project_context_pack(tmp.path()).expect("pack");
        assert!(pack.contains("README.md"), "README must survive truncation");
        assert!(
            pack.contains("Cargo.toml"),
            "config file must survive truncation"
        );
    }
}
