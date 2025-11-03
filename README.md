# 🎨 Air Canvas – Draw with Your Hand using AI (Mediapipe + OpenCV)

> **Author:** Jemish Koladiya  
> **Tech Stack:** Python · MediaPipe · OpenCV · NumPy  

---

## 🧠 Overview

**Air Canvas** is an AI-powered virtual drawing application that lets you **draw on your screen without touching anything** just by moving your **index finger in the air** ✋.  

Built with **MediaPipe Hands** for real-time hand detection and **OpenCV** for visual rendering, it intelligently tracks your finger and converts your motion into colorful, smooth digital strokes.

---

## ✨ Features

| Feature | Description |
|----------|-------------|
| 🖐️ **AI Hand Tracking** | Tracks your hand and finger landmarks in real time using Google’s MediaPipe model. |
| ✏️ **Draw with One Finger** | When one finger is up → you draw; when more are up → you pause. |
| 🎨 **Color Palette Toolbar** | Select from Blue, Green, Red, or Yellow colors using on-screen buttons. |
| 💾 **Save Your Drawing** | Instantly save your artwork as a `.png` file with timestamp. |
| 🧹 **Clear Canvas** | Wipe the canvas clean with a single gesture. |
| 🖼️ **Beautiful UI** | Smooth toolbar design, live color indicators, and real-time feedback panel. |
| 🔢 **Status Panel** | Displays current mode (Drawing / Paused), selected color, and finger count. |
| 🧮 **Mathematical Smoothing** | Uses interpolation to create fluid, natural lines between frames. |

---

## 🏗️ Tech Stack

- **Python 3.8+**
- **MediaPipe (Hands Solution)** → AI hand detection and 21 landmark prediction  
- **OpenCV** → Frame rendering, color detection, and canvas visualization  
- **NumPy** → Image array processing  
- **Deque (collections)** → Efficient storage for drawing coordinates  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/air-canvas.git
cd air-canvas
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the script
```bash
python airCanvas.py
```