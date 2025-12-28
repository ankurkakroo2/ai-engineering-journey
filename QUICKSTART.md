# Quick Start Guide

Get started with your 8-week AI engineering journey in 5 minutes!

## Step 1: Review the Plan (5 min)

```bash
cd /Users/ankur/D/Playground/ai-engineering-journey
```

Read these files in order:
1. **README.md** - Overview and navigation
2. **PLAN.md** - Complete 8-week curriculum
3. **week-01/README.md** - This week's detailed plan

## Step 2: Set Up Your Environment (10 min)

### Create API Keys
You'll need these for Week 1:

1. **OpenAI API Key**
   - Go to https://platform.openai.com/api-keys
   - Create new key
   - Save it somewhere safe

2. **Anthropic (Claude) API Key**
   - Go to https://console.anthropic.com/
   - Create API key
   - You're already using Claude Code, so you may have this!

### Environment Setup
```bash
# Create .env file in the root
cat > .env << EOF
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
EOF

# Install essential tools (if not already installed)
# Python
brew install python@3.11  # or use pyenv

# Node (for later weeks)
brew install node

# Docker (for databases)
brew install --cask docker
```

## Step 3: Start Week 1 (Today!)

### Day 1-2: World View Building

**Reading List** (6-8 hours over 2 days):
1. [What is Vector Search?](https://www.pinecone.io/learn/vector-search/) - 30 min
2. [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) - 45 min
3. [Andrej Karpathy: Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g) - 1 hour
4. Browse: [Claude Documentation](https://docs.anthropic.com/) - 1 hour
5. Explore: ChromaDB basics - 1 hour

**Exercise**:
Create a mental model diagram answering:
- What are embeddings?
- How do vector databases work?
- What is semantic search?
- How does it all fit together?

Save your diagram in `week-01/notes/mental-model.md`

### Day 3: Start Building

```bash
# Navigate to week 1
cd week-01/project

# Create your first project
mkdir semantic-code-search
cd semantic-code-search

# Initialize Python project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install openai chromadb click rich python-dotenv

# Create basic structure
touch main.py
touch requirements.txt
touch README.md

# Use Claude Code to help you build!
```

**First Task**: Build a simple script that:
1. Takes a text input
2. Generates an embedding using OpenAI
3. Stores it in ChromaDB
4. Retrieves similar texts

## Step 4: Daily Workflow

### Morning Routine (30 min)
1. Open `week-01/notes/daily-log.md`
2. Review today's goals
3. Read one article/resource
4. Plan your coding session

### Coding Session (2-4 hours)
1. Use Claude Code to help you build
2. But write the core logic yourself
3. Test as you go
4. Commit frequently to git

### Evening Wrap-Up (15 min)
1. Update daily log with what you learned
2. Commit your code
3. Post on LinkedIn/Twitter (optional but recommended)
4. Plan tomorrow

## Step 5: Leverage Claude Code Effectively

### How to Use Claude Code

**For Learning**:
```
"Explain how cosine similarity works in semantic search"
"What are the trade-offs between different chunking strategies?"
"Help me understand this error: [paste error]"
```

**For Building**:
```
"Help me scaffold a Click CLI application"
"Review this code for best practices: [paste code]"
"Generate tests for this function: [paste function]"
```

**For Debugging**:
```
"This search isn't returning relevant results. Help me debug."
"How can I optimize this embedding pipeline?"
"What's wrong with my ChromaDB query?"
```

### Golden Rule
- Let Claude help with scaffolding, debugging, and best practices
- But understand every line of code
- Write the core logic yourself to learn deeply

## Week 1 Checklist

### Setup ✅
- [ ] Environment configured
- [ ] API keys obtained
- [ ] Development tools installed
- [ ] Git repository initialized

### Learning ✅
- [ ] Read vector search guide
- [ ] Read embeddings documentation
- [ ] Watched Karpathy video
- [ ] Created mental model

### Building ✅
- [ ] Project scaffolded
- [ ] Basic embedding pipeline working
- [ ] ChromaDB integration complete
- [ ] Search functionality implemented
- [ ] CLI interface built

### Shipping ✅
- [ ] Code committed to GitHub
- [ ] Documentation written
- [ ] Blog post drafted
- [ ] Shared on social media

## Tips for Success

### Time Management
- **Block time**: Dedicate specific hours each day
- **Eliminate distractions**: Close Slack, email, etc.
- **Take breaks**: Pomodoro technique works well
- **Don't perfectionism**: Ship and iterate

### Learning Approach
- **Build to learn**: Don't just read, code
- **Explain to understand**: Write blog posts
- **Ask questions**: Use Claude Code actively
- **Reflect daily**: What did you learn?

### Staying Motivated
- **Share progress**: Post daily updates
- **Celebrate wins**: Shipped a feature? Tweet it!
- **Join communities**: Discord servers, Reddit
- **Visualize the goal**: Your portfolio in 8 weeks

## Common Pitfalls to Avoid

❌ **Tutorials hell**: Don't just watch, build
❌ **Perfectionism**: Ship MVP, iterate later
❌ **Scope creep**: Stick to the week's project
❌ **Isolation**: Share your journey, ask for help
❌ **Burnout**: Take breaks, pace yourself

✅ **Do this instead**: Build → Ship → Learn → Repeat

## Getting Help

### When Stuck
1. **Google it** - You're not the first!
2. **Ask Claude Code** - Your AI pair programmer
3. **Check documentation** - Primary source of truth
4. **Stack Overflow** - Community wisdom
5. **Discord/Reddit** - Active communities

### Resources
- See `resources/README.md` for comprehensive links
- Check Week 1 README for week-specific resources
- Use PROGRESS.md to track your journey

## Ready to Start?

1. **Right now**: Read Pinecone's vector search guide
2. **Today**: Complete Day 1 reading
3. **This week**: Ship your first AI tool
4. **Next 8 weeks**: Transform into an AI engineer

---

**Remember**: You're a Director of Engineering with strong systems thinking. You've got this! The goal isn't perfection—it's progress, learning, and shipping.

**Let's build!** 🚀

Start with: `week-01/README.md`
