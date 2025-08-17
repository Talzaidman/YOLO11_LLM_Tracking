# 🎯 Text-Guided Real-Time Video Attention and ROI Detection  

This project implements a **real-time, text-guided video attention system** that integrates deep learning object detection (YOLOv8-World) with **dynamic camera ROI (Region of Interest) control**.  
The system enables efficient high-speed tracking by adjusting the camera’s ROI in real time, reducing computational load while maintaining accuracy. It also leverages an **LLM-based interface** to allow users to specify targets through natural language commands.  

---

## ✨ Features  
- 📝 **Text-guided detection** using natural language commands (via LLM).  
- 🚀 **Dynamic ROI control** for improved FPS and reduced aliasing.  
- 👀 **Periodic full-frame updates** to maintain spatial awareness.  
- ⚡ **Multi-threaded architecture** for low-latency performance.  
- 🧪 **Simulation-tested** on controlled scenarios (fan and walking man video).  

---

## ⚙️ Hardware Requirements  
- **Camera:** Allied Vision Manta (controlled via Vimba SDK).  
- **GPU:** NVIDIA GPU recommended for YOLOv8-World inference but not necessarily. the system was tested with an AMD Radeon   
- **Workstation:** Multi-core CPU and ≥16 GB RAM suggested.  

---

## 💻 Software Requirements  
- **Python:** 3.9+  
- **Dependencies:**  
  - [Ultralytics YOLOv8](https://docs.ultralytics.com/)  
  - [OpenCV](https://opencv.org/)  
  - [NumPy](https://numpy.org/)  
  - [Matplotlib](https://matplotlib.org/)  
  - [Vimba SDK](https://www.alliedvision.com/en/products/software/vimba-sdk/)  

Install them via:  
```bash
pip install -r requirements.txt
