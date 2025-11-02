# Deployment & Collaboration Guide

## 📥 Clone on Other Machines

### Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/Maryam-bigonah/energy-management-system-.git

# Or using SSH (if you have SSH keys set up)
git clone git@github.com:Maryam-bigonah/energy-management-system-.git

# Navigate to the project
cd energy-management-system-
```

### Setup on New Machine

#### 1. Install Dependencies

**Backend (Python):**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend (Node.js):**
```bash
cd frontend
npm install
```

#### 2. Run the Application

**Option 1: Using Docker (Recommended)**
```bash
docker-compose up --build
```

**Option 2: Manual Start**
```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm start
```

**Option 3: Using Start Scripts**
```bash
# Linux/Mac
./start.sh

# Windows
start.bat
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

---

## 👥 Share with Collaborators

### Add Collaborators to GitHub Repository

1. Go to your repository: https://github.com/Maryam-bigonah/energy-management-system-
2. Click **Settings** → **Collaborators** → **Add people**
3. Enter their GitHub username or email
4. They will receive an invitation email

### For Collaborators - Getting Started

After being added as a collaborator:

1. **Clone the repository:**
   ```bash
   git clone git@github.com:Maryam-bigonah/energy-management-system-.git
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and commit:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

4. **Push to GitHub:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub to merge changes

### Branch Strategy

- `main` - Production-ready code
- `develop` - Development branch (optional)
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

---

## 🚀 Deploy to Servers

### Option 1: Deploy with Docker (Recommended)

#### On Your Server:

1. **Install Docker and Docker Compose:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io docker-compose

   # macOS (via Homebrew)
   brew install docker docker-compose
   ```

2. **Clone repository:**
   ```bash
   git clone https://github.com/Maryam-bigonah/energy-management-system-.git
   cd energy-management-system-
   ```

3. **Configure environment variables:**
   ```bash
   # Edit docker-compose.yml if needed
   # Set REACT_APP_API_URL to your server's backend URL
   ```

4. **Build and start:**
   ```bash
   docker-compose up -d --build
   ```

5. **Check status:**
   ```bash
   docker-compose ps
   docker-compose logs
   ```

6. **Stop services:**
   ```bash
   docker-compose down
   ```

### Option 2: Deploy to Cloud Platforms

#### Heroku Deployment

**Backend (Flask):**
```bash
# Install Heroku CLI
heroku login
heroku create your-app-name-backend
cd backend
git subtree push --prefix backend heroku main
```

**Frontend (React):**
```bash
cd frontend
npm run build
# Deploy build folder to hosting service (Netlify, Vercel, etc.)
```

#### AWS/Azure/GCP

Use their container services (ECS, Container Instances, Cloud Run) with Docker images.

### Production Checklist

- [ ] Set `FLASK_ENV=production` in backend
- [ ] Set `REACT_APP_API_URL` to production API URL
- [ ] Enable HTTPS/SSL certificates
- [ ] Set up environment variables securely
- [ ] Configure database (if using one)
- [ ] Set up monitoring and logging
- [ ] Configure CORS for production domain
- [ ] Set up backup strategy for data

---

## 📊 Track Changes and Versions

### Git Workflow

#### Check Status
```bash
git status
```

#### See Changes
```bash
# See what changed
git diff

# See commit history
git log

# See commit history with graph
git log --oneline --graph --all
```

#### Create a Version Tag

**Semantic Versioning:**
- Major version: `v1.0.0` (breaking changes)
- Minor version: `v1.1.0` (new features)
- Patch version: `v1.0.1` (bug fixes)

```bash
# Create and push a tag
git tag -a v1.0.0 -m "Initial LSTM forecasting system"
git push origin v1.0.0

# List tags
git tag

# Checkout specific version
git checkout v1.0.0
```

### Release Management

#### Create a Release on GitHub

1. Go to repository → **Releases** → **Create a new release**
2. Choose tag (or create new one)
3. Add release notes
4. Publish release

#### Release Notes Template

```markdown
## Release v1.0.0

### Added
- LSTM energy forecasting model
- Master dataset builder
- PVGIS API integration
- Full-stack web application

### Changed
- Updated frontend components
- Improved data loaders

### Fixed
- Coordinate detection in building CSV
- Data loader array operations
```

### Branch Protection (Optional)

To protect `main` branch:
1. Settings → Branches → Add rule
2. Require pull request reviews
3. Require status checks
4. Require branches to be up to date

---

## 🔄 Common Git Commands

### Daily Workflow

```bash
# Pull latest changes
git pull origin main

# Check what changed
git status
git diff

# Stage changes
git add .
git add specific-file.py

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push origin main

# Create new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# Merge branch
git checkout main
git merge feature/new-feature
git push origin main
```

### Undo Changes

```bash
# Undo unstaged changes
git checkout -- filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert a commit (creates new commit)
git revert <commit-hash>
```

---

## 🔐 Security Best Practices

1. **Never commit sensitive data:**
   - API keys
   - Passwords
   - Database credentials
   - Use `.env` files (already in `.gitignore`)

2. **Use environment variables:**
   ```bash
   # Create .env file (not committed to git)
   FLASK_SECRET_KEY=your-secret-key
   API_KEY=your-api-key
   ```

3. **Review `.gitignore`:**
   - Make sure sensitive files are ignored
   - Check before committing

---

## 📚 Additional Resources

- **GitHub Docs**: https://docs.github.com
- **Docker Docs**: https://docs.docker.com
- **React Deployment**: https://create-react-app.dev/docs/deployment/
- **Flask Deployment**: https://flask.palletsprojects.com/en/latest/deploying/

---

## 🆘 Troubleshooting

### Clone Issues
```bash
# If clone fails, try:
git clone --depth 1 https://github.com/Maryam-bigonah/energy-management-system-.git
```

### Merge Conflicts
```bash
# See conflicts
git status

# Resolve conflicts, then:
git add .
git commit -m "Resolve merge conflicts"
```

### Push Rejected
```bash
# Pull latest first
git pull origin main

# Then push
git push origin main
```

### Docker Issues
```bash
# Clean Docker cache
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

---

Your repository is ready for collaboration and deployment! 🚀

