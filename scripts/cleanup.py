#!/usr/bin/env python3
"""
Cleanup script for SAGED-Bias repository.
Removes build artifacts, cache files, and temporary files.
"""

import os
import shutil
import glob
import argparse
from pathlib import Path

def remove_patterns(patterns, description, dry_run=False):
    """Remove files/directories matching the given patterns."""
    removed_count = 0
    total_size = 0
    
    print(f"\n🔍 {description}...")
    
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            path = Path(match)
            if path.exists():
                try:
                    if path.is_file():
                        size = path.stat().st_size
                        total_size += size
                        if not dry_run:
                            path.unlink()
                        print(f"  {'[DRY RUN] ' if dry_run else ''}Removed file: {match} ({size} bytes)")
                    elif path.is_dir():
                        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        total_size += size
                        if not dry_run:
                            shutil.rmtree(match)
                        print(f"  {'[DRY RUN] ' if dry_run else ''}Removed directory: {match} ({size} bytes)")
                    removed_count += 1
                except (OSError, PermissionError) as e:
                    print(f"  ❌ Failed to remove {match}: {e}")
    
    if removed_count > 0:
        print(f"  ✅ {'Would remove' if dry_run else 'Removed'} {removed_count} items ({total_size:,} bytes)")
    else:
        print(f"  ℹ️  No items found to remove")
    
    return removed_count, total_size

def main():
    parser = argparse.ArgumentParser(description="Clean up SAGED-Bias repository")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")
    parser.add_argument("--all", action="store_true", help="Remove all temporary files including coverage reports")
    
    args = parser.parse_args()
    
    print("🧹 SAGED-Bias Repository Cleanup")
    print("=" * 40)
    
    # Define cleanup patterns
    cleanup_patterns = {
        "Python cache files": [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/*.pyd",
        ],
        "Build artifacts": [
            "build/",
            "dist/",
            "*.egg-info/",
            ".eggs/",
        ],
        "Test artifacts": [
            ".pytest_cache/",
            ".coverage*",
        ],
        "IDE files": [
            ".vscode/settings.json",
            ".idea/workspace.xml",
            "*.swp",
            "*.swo",
            "*~",
        ],
        "Temporary files": [
            "*.tmp",
            "*.temp",
            ".DS_Store",
            "Thumbs.db",
        ]
    }
    
    if args.all:
        cleanup_patterns["Coverage reports"] = ["htmlcov/"]
        cleanup_patterns["Legacy files"] = ["poetry.lock"]
    
    total_removed = 0
    total_size = 0
    
    for description, patterns in cleanup_patterns.items():
        count, size = remove_patterns(patterns, description, args.dry_run)
        total_removed += count
        total_size += size
    
    print("\n" + "=" * 40)
    if args.dry_run:
        print(f"🔍 Dry run complete: Would remove {total_removed} items ({total_size:,} bytes)")
        print("Run without --dry-run to actually remove files")
    else:
        print(f"✅ Cleanup complete: Removed {total_removed} items ({total_size:,} bytes)")
    
    if total_removed == 0:
        print("🎉 Repository is already clean!")

if __name__ == "__main__":
    main() 