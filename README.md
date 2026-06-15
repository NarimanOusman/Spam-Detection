# Spam Detection Web Application

A beginner-friendly **MLOps demo project**. This project demonstrates the full lifecycle of a machine learning application — from training a model to deploying it on the cloud.

>  Follow this README step by step. By the end, you will have your own spam detector running on the internet!

---

##  What You Will Build

A web app that classifies text messages as **Spam** or **Not Spam** using machine learning.

| Component | Technology |
|---|---|
| Machine Learning | Multinomial Naive Bayes |
| Feature Extraction | CountVectorizer (scikit-learn) |
| Web Framework | Flask (Python) |
| Frontend | HTML5 + CSS3 |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Option A | AWS EC2 |
| Cloud Option B | Azure Web Apps |

---

## Project Structure

```
Spam-Detection/
│
├── app.py                   # Flask web application (runs the website)
├── train.py                 # Trains the ML model and saves it
├── model.pkl                # Trained model file (created after running train.py)
├── vectorizer.pkl           # Text vectorizer file (created after running train.py)
├── requirements.txt         # Python packages needed
├── Dockerfile               # Instructions to build a Docker container
├── .gitignore               # Files Git should ignore
├── README.md                # This file
│
├── templates/
│   └── index.html           # The web page (HTML + CSS)
│
└── .github/
    └── workflows/
        └── docker-build.yml # Automatic Docker build & push (GitHub Actions)
```

---

## PART 1 — Clone & Run Locally

These steps get the project running on your own computer.

---

### Prerequisites

Make sure you have these installed before starting:

| Tool | How to check | Download |
|---|---|---|
| Python 3.9+ | `python3 --version` | https://www.python.org |
| pip | `pip --version` | Comes with Python |
| Git | `git --version` | https://git-scm.com |

---

### Step 1 — Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/sidicksino/Spam-Detection.git
cd Spam-Detection
```

You should now see the project files in your terminal.

---

### Step 2 — The Dataset is Already Included

The `spam.csv` file is included in the repository, so you get it automatically when you clone. Nothing extra to do!

> The dataset contains **5,572 SMS messages** labelled as `ham` (legitimate) or `spam`.

---


### Step 3 — Create a Virtual Environment

A virtual environment keeps your project's packages separate from the rest of your computer.

```bash
# Create the virtual environment
python3 -m venv venv
```

Now activate it:

```bash
# On Mac / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

> You will see `(venv)` appear at the start of your terminal line. That means it worked!

---

### Step 4 — Install the Required Packages

```bash
pip install -r requirements.txt
```

```bash
# Or
python -m pip install -r requirements.txt
```

This installs Flask, scikit-learn, pandas, and gunicorn. Wait for it to finish.

---

### Step 5 — Train the Machine Learning Model

```bash
python train.py
```

This will:
1. Load `spam.csv`
2. Clean and preprocess the text
3. Train a Naive Bayes model
4. Print the accuracy and evaluation metrics
5. Save `model.pkl` and `vectorizer.pkl`

**Expected output:**
```
--- PART 1: DATA PREPROCESSING ---
Loading dataset from spam.csv...
Dataset loaded successfully with 5572 rows.
No missing values found.
Cleaning text messages...
Splitting dataset: 80% train / 20% test
Training samples: 4456
Testing samples: 1114

--- PART 2: MODEL TRAINING & EVALUATION ---
Training Multinomial Naive Bayes model...
Model Accuracy: 0.9856 (98.56%)

Confusion Matrix:
[[967   6]
 [ 10 131]]

Saving trained model to 'model.pkl'...
Saving CountVectorizer to 'vectorizer.pkl'...
Training and exporting complete!
```

After running, you will see two new files: `model.pkl` and `vectorizer.pkl`.

---

### Step 6 — Run the Web Application

```bash
python app.py
```

Open your browser and go to:
```
http://127.0.0.1:5000
```

Type a message and click **Predict**! Try these examples:

- **Spam:** `Congratulations! You've won a FREE iPhone. Click here to claim your prize NOW!`
- **Not Spam:** `Hey, are you coming to class tomorrow?`

Press `CTRL + C` in the terminal to stop the app.

---

## PART 2 — Push to Your Own GitHub Repository

When you clone this repository, git automatically connects your local folder to the **instructor's** GitHub account. You need to change that connection to point to **your own** GitHub account before you can push.

> **Important:** Do NOT run `git init`. The `.git` folder already exists from the clone. You just need to change where it pushes to.

---

### Step 1 — Create a New GitHub Repository

1. Go to [github.com](https://github.com) and log in
2. Click the **+** button → **New repository**
3. Name it: `Spam-Detection`
4. Set it to **Public**
5. **Do NOT** check "Initialize with README" (we already have one)
6. Click **Create repository**

---

### Step 2 — Disconnect from the Instructor's Repo

When you cloned this project, git set the remote `origin` to point to the instructor's GitHub. Run this command to remove it:

```bash
git remote remove origin
```

You can verify it's gone with:
```bash
git remote -v
# Should show nothing
```

---

### Step 3 — Connect to Your Own Repo

Now link your local folder to the GitHub repository you just created (replace `YOUR_GITHUB_USERNAME`):

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/Spam-Detection.git
```

Verify it's set correctly:
```bash
git remote -v
# Should show your own GitHub URL
```

---

### Step 4 — Commit and Push

```bash
# Stage all files
git add .

# Commit with a message
git commit -m "My spam detection project"

# Make sure branch is named main
git branch -M main

# Push to your GitHub
git push -u origin main
```

> GitHub may ask for your username and password. Use your GitHub username and a **Personal Access Token** (not your regular password). Create one at: **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)**

---

### Step 5 — Verify

Go to `https://github.com/YOUR_GITHUB_USERNAME/Spam-Detection` — you should see all your files there!

---


## PART 3 — Docker

Docker packages your app so it can run on any computer or server.

---

### Step 1 — Install Docker

Download and install Docker Desktop: https://www.docker.com/products/docker-desktop

Verify it works:
```bash
docker --version
```

---

### Step 2 — Build the Docker Image

Make sure you are inside the project folder, then run:

```bash
docker build -t spam-detector .
```

> This reads the `Dockerfile` and creates an image. It may take a few minutes the first time.

---

### Step 3 — Run the Container

```bash
docker run -p 5000:5000 spam-detector
```

Open your browser at: **http://localhost:5000**

To stop the container, press `CTRL + C`.

---

### Step 4 — Create a Docker Hub Account

1. Go to [hub.docker.com](https://hub.docker.com) and create a free account
2. Note your Docker Hub username (you will need it below)

---

### Step 5 — Push Your Image to Docker Hub

```bash
# Log in to Docker Hub
docker login

# Tag your image (replace YOUR_DOCKERHUB_USERNAME)
docker tag spam-detector YOUR_DOCKERHUB_USERNAME/spam-detector:latest

# Push the image
docker push YOUR_DOCKERHUB_USERNAME/spam-detector:latest
```

Your image is now publicly available on Docker Hub!

---

## PART 4 — GitHub Actions (Automatic CI/CD)

Every time you push code to GitHub, GitHub Actions will automatically build and push your Docker image. No manual steps needed!

---

### Step 1 — Add Your Docker Credentials as GitHub Secrets

In your GitHub repository:
1. Click **Settings** (top menu)
2. Click **Secrets and variables** → **Actions**
3. Click **New repository secret** and add:

| Secret Name | Value |
|---|---|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub password or access token |

---

### Step 2 — Push Any Change to Trigger the Workflow

```bash
# Make a small change, for example edit a comment in app.py, then:
git add .
git commit -m "Trigger CI/CD pipeline"
git push origin main
```

---

### Step 3 — Watch It Run

1. Go to your GitHub repository
2. Click the **Actions** tab
3. You will see your workflow running — click it to watch the logs!

When it finishes with a green , your Docker image has been automatically updated on Docker Hub.

---

## PART 5 — Deploy to AWS (EC2)

Run your Docker container on a cloud server so anyone in the world can access it.

---

### Step 1 — Create an AWS Account

Go to [aws.amazon.com](https://aws.amazon.com) and create a free account (credit card required but free tier available).

---

### Step 2 — Launch an EC2 Instance

1. Go to **AWS Console** → search for **EC2** → click **Launch Instance**
2. Fill in the settings:
   - **Name:** `spam-detector-server`
   - **AMI (OS):** `Ubuntu Server 22.04 LTS` (Free Tier eligible)
   - **Instance type:** `t2.micro` (Free Tier eligible)
   - **Key pair:** Click **Create new key pair** → name it `my-key` → download the `.pem` file → **keep it safe!**
3. Under **Network settings** → Edit → Add inbound rule:
   - Type: **Custom TCP** | Port: **5000** | Source: **0.0.0.0/0**
4. Click **Launch Instance**
5. Wait about 1 minute for it to start

---

### Step 3 — Connect to Your Server via SSH

```bash
# Set the correct permissions on your key file (Mac/Linux only)
chmod 400 my-key.pem

# Connect to the server (replace YOUR_EC2_PUBLIC_IP with the IP shown in AWS)
ssh -i my-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

> Find your EC2 Public IP in the AWS Console → EC2 → Instances → click your instance → look for **Public IPv4 address**

---

### Step 4 — Install Docker on the Server

You are now inside the Ubuntu server. Run these commands one by one:

```bash
# Update the package list
sudo apt update

```bash
# Install Docker
sudo apt install docker.io -y
```

```bash
# Verify Docker
docker --version
```

```bash
# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
```

---

### Step 5 — Pull and Run Your Docker Image

```bash
# Pull your image from Docker Hub (replace YOUR_DOCKERHUB_USERNAME)
docker pull YOUR_DOCKERHUB_USERNAME/spam-detector:latest

# Run the container in the background
docker run -d -p 5000:5000 --name spam-app YOUR_DOCKERHUB_USERNAME/spam-detector:latest
```

---

### Step 6 — Open in Browser

Go to:
```
http://YOUR_EC2_PUBLIC_IP:5000
```

Your app is live on the internet!

---

## PART 6 — Deploy to Azure (Web Apps)

A simpler cloud deployment — no SSH needed!

---

### Step 1 — Create an Azure Account

Go to [azure.microsoft.com](https://azure.microsoft.com) and create a free account (students get $100 credit with Azure for Students).

---

### Step 2 — Create a Web App

1. Go to [portal.azure.com](https://portal.azure.com) and sign in
2. Click **+ Create a resource** → search **Web App** → click **Create**
3. Fill in:
   - **Resource Group:** click **Create new** → name it `spam-rg`
   - **Name:** `spam-detector-yourname` *(must be globally unique)*
   - **Publish:** **Docker Container** ← very important!
   - **Operating System:** Linux
   - **Region:** Pick the one closest to you
4. Click **Next: Docker →**

---

### Step 3 — Connect to Docker Hub

1. **Image Source:** Docker Hub
2. **Access Type:** Public
3. **Image and tag:** `YOUR_DOCKERHUB_USERNAME/spam-detector:latest`
4. Click **Review + Create** → **Create**

---

### Step 4 — Set the Port

After deployment:
1. Go to your Web App → **Configuration** → **Application settings**
2. Click **+ New application setting**
3. Name: `WEBSITES_PORT` | Value: `5000`
4. Click **Save**

---

### Step 5 — Access Your App

1. Go to **Overview** → look for **Default domain**
2. Your URL looks like:
```
https://spam-detector-yourname.azurewebsites.net
```

Open it in your browser — your app is live!

---

##  Dataset Information

- **Name:** SMS Spam Collection Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Size:** 5,572 messages — 4,825 legitimate (ham) + 747 spam
- **Format:** CSV with two columns: `label` and `message`

---

## License

This project is for educational purposes. Dataset credit: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
