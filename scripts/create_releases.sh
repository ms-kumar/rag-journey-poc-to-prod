#!/bin/bash
# Script to create and push all release tags
# Run this to publish all weekly releases to GitHub

set -e

echo "🏷️  Creating release tags for all weeks..."
echo ""

# Define releases (tag, date, title)
declare -a releases=(
    "v0.1.0|2025-11-30|Week 1: Naive RAG Pipeline"
    "v0.2.0|2025-12-07|Week 2: Production-Ready RAG Enhancements"
    "v0.3.0|2025-12-14|Week 3: Hybrid Retrieval with Fusion"
    "v0.4.0|2025-12-21|Week 4: Metadata Filtering & Query Refinement"
    "v0.5.0|2025-12-28|Week 5: Evaluations & Guardrails"
    "v0.6.0|2026-01-04|Week 6: Schema Consolidation & Architectural Refinement"
    "v0.7.0|2026-01-11|Week 7: Agentic RAG Implementation"
    "v0.8.0|2026-01-18|Week 8: Observability, Experimentation & Production Readiness"
)

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes"
    echo "Please commit or stash them before creating releases"
    exit 1
fi

echo "Creating annotated tags..."
echo ""

for release in "${releases[@]}"; do
    IFS='|' read -r tag date title <<< "$release"
    
    # Check if tag already exists
    if git rev-parse "$tag" >/dev/null 2>&1; then
        echo "⏭️  Tag $tag already exists, skipping..."
    else
        echo "📝 Creating tag: $tag ($title)"
        
        # Create annotated tag with date
        GIT_COMMITTER_DATE="$date 12:00:00" \
        git tag -a "$tag" -m "$title" -m "Release date: $date"
        
        echo "✅ Created tag: $tag"
    fi
done

echo ""
echo "📊 All tags created. Current tags:"
git tag --sort=-version:refname

echo ""
echo "🚀 To push all tags to GitHub, run:"
echo "   git push origin --tags"
echo ""
echo "This will trigger the release workflow for each tag!"
echo ""
echo "⚠️  Note: Make sure you've pushed all commits first:"
echo "   git push origin main"
