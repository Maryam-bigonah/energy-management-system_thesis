# How to View Your Repository and Files

## 🌐 View on GitHub (Web Browser)

### 1. View Your Repository Online

**Direct Link:**
```
https://github.com/Maryam-bigonah/energy-management-system-
```

**Steps:**
1. Open your web browser
2. Go to: `https://github.com/Maryam-bigonah/energy-management-system-`
3. You'll see all your files, commits, and documentation

### 2. View Specific Files

**In the repository:**
- Click on any file to view its contents
- Click on folders to navigate
- Use the search bar (press `/`) to find files quickly

**View Documentation:**
- `README.md` - Main project documentation
- `DEPLOYMENT.md` - Deployment guide
- `QUICK_START.md` - Quick start guide
- `TEST_RESULTS.md` - Test results
- `.github/README_COLLABORATION.md` - Collaboration guide

### 3. View File History

- Click on any file
- Click **History** button to see all changes
- Click on a commit to see what changed

### 4. View Commits

- Click **Commits** in the repository
- See all commit messages and changes
- Click on any commit to see details

---

## 💻 View Locally on Your Computer

### View Files in Your Current Location

**You already have the files here:**
```
/Users/mariabigonah/Desktop/thesis/code/
```

### View Files Using Terminal

```bash
# Navigate to your project
cd /Users/mariabigonah/Desktop/thesis/code

# List all files
ls -la

# View a specific file
cat README.md
cat DEPLOYMENT.md

# View file in a text editor
open README.md          # macOS
code README.md          # VS Code
```

### View Files in Finder (macOS)

1. Open Finder
2. Navigate to: `/Users/mariabigonah/Desktop/thesis/code`
3. You'll see all your project files
4. Double-click any file to open it

### View Files in VS Code / Cursor

```bash
# Open in VS Code
code /Users/mariabigonah/Desktop/thesis/code

# Or open in Cursor
cursor /Users/mariabigonah/Desktop/thesis/code
```

---

## 📱 Quick View Commands

### View Project Structure

```bash
# Tree view (if you have tree installed)
tree -L 2

# Or use find
find . -maxdepth 2 -type f -name "*.py" -o -name "*.md"
```

### View Key Files

```bash
# Main documentation
cat README.md

# Deployment guide
cat DEPLOYMENT.md

# Test results
cat TEST_RESULTS.md

# Quick start
cat QUICK_START.md
```

### View Code Files

```bash
# LSTM model
cat lstm_energy_forecast.py

# Master dataset builder
cat build_master_dataset.py

# Backend API
cat backend/app.py
```

---

## 🔍 View on GitHub - Step by Step

### 1. Open GitHub in Browser

Go to: **https://github.com**

### 2. Sign In

Use your GitHub credentials

### 3. Navigate to Your Repository

- Click your profile icon (top right)
- Click **Your repositories**
- Click **energy-management-system-**

OR go directly to:
**https://github.com/Maryam-bigonah/energy-management-system-**

### 4. Explore the Repository

**Main Page Shows:**
- File list
- README.md (if exists)
- Recent commits
- Branch information

**Navigation:**
- **Code** - View all files
- **Issues** - Track issues
- **Pull requests** - Review changes
- **Actions** - CI/CD (if configured)
- **Settings** - Repository settings

### 5. View Specific Sections

**View Documentation:**
1. Click `README.md` or `DEPLOYMENT.md`
2. Files open in GitHub's viewer
3. See formatted markdown

**View Code:**
1. Click any `.py` or `.js` file
2. See syntax-highlighted code
3. Click line numbers to get permalink

**View Recent Changes:**
1. Click **Commits** link
2. See commit history
3. Click any commit to see changes

---

## 📊 View Repository Statistics

### On GitHub:

**Insights Tab:**
- Go to repository → **Insights**
- See:
  - Contributors
  - Commits (graph)
  - Code frequency
  - Traffic

**Code Tab:**
- See file tree
- Search files (`t` key)
- Browse folders

---

## 🚀 View Running Application

### Local View

**After starting the app:**

```bash
# Start the application
docker-compose up
# OR
cd backend && python app.py  # Terminal 1
cd frontend && npm start     # Terminal 2
```

**Then open browser:**
- Frontend: **http://localhost:3000**
- Backend API: **http://localhost:5000**
- API Docs: **http://localhost:5000/api/health**

### View Application Logs

```bash
# Docker logs
docker-compose logs -f

# Backend logs (if running manually)
# Will show in terminal where you ran python app.py

# Frontend logs (if running manually)
# Will show in terminal where you ran npm start
```

---

## 📖 Quick Reference Links

### GitHub Repository
```
https://github.com/Maryam-bigonah/energy-management-system-
```

### Local Path
```
/Users/mariabigonah/Desktop/thesis/code/
```

### Key Documentation Files
- `README.md` - Main documentation
- `DEPLOYMENT.md` - How to deploy
- `QUICK_START.md` - Quick start guide
- `TEST_RESULTS.md` - Test information
- `HOW_TO_VIEW.md` - This file!

---

## 💡 Tips

1. **Bookmark your repository URL** for quick access
2. **Use GitHub's search** (`/` key) to find files quickly
3. **Use VS Code/Cursor** to view files with syntax highlighting
4. **Use `git log`** in terminal to see commit history
5. **Use GitHub Desktop** (GUI) if you prefer visual interface

---

Need to see something specific? Let me know what you'd like to view! 👀

