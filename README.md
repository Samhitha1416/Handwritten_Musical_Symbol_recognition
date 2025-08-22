# Handwritten_Musical_Symbol_recognition

# 🎼 CNN-Based Approach for Generating Musical Symbols and Sounds from Handwritten Input  

## 📌 Overview  
This project implements a **Convolutional Neural Network (CNN)-based system** that recognizes **handwritten musical notes** and generates the corresponding **sound output**. The system bridges the gap between **visual music notation** and **auditory learning**, making it useful for music learners, educators, and digital preservation of handwritten scores.  

Users can upload or draw handwritten music notes, and the system will:  
1. Recognize the note using a trained CNN model.  
2. Map the note to its corresponding pitch and duration.  
3. Generate and play the sound using MIDI/audio libraries.  

---

## 🚀 Features  
- Handwritten **musical note recognition** using CNNs.  
- **Real-time classification** of musical symbols.  
- **Audio playback** of recognized notes (MIDI-based).  
- User-friendly **Tkinter GUI** for interaction.  
- Supports **common notes**: Whole, Half, Quarter, Eighth, and Sixteenth.  

---

## 🏗️ System Architecture  
1. **Preprocessing** – Grayscale conversion, resizing, noise removal, normalization.  
2. **CNN Model** – Extracts features and classifies handwritten musical notes.  
3. **Symbol Mapping** – Converts recognized notes into digital music representation.  
4. **Sound Generation** – Plays audio output using MIDI/pydub libraries.  
5. **GUI Interface** – Allows uploading handwritten input and playing results.  

---

## 🖥️ Tech Stack  
- **Language**: Python  
- **Deep Learning**: TensorFlow/Keras (or PyTorch)  
- **GUI Framework**: Tkinter  
- **Audio Libraries**: MIDI / pydub  
- **IDE**: Visual Studio Code  

---

## ⚙️ Requirements  

### Software  
- Python 3.8+  
- TensorFlow/Keras  
- Tkinter  
- pydub / MIDI libraries  

### Hardware
-Processor: Intel i3 or above (1.5 GHz+)
-RAM: 2 GB+
-Disk: 250 GB+

## 📊 Results  

The developed system successfully recognized handwritten musical notes and generated their corresponding audio using a trained CNN model. The Tkinter GUI allowed users to upload handwritten inputs and receive both **visual recognition** and **sound playback** in real time.  

### ✅ Key Outcomes  
- Accurate recognition of handwritten notes across varied handwriting styles.  
- Integrated **visual output** (note classification) and **audio playback** (MIDI-based).  
- Interactive GUI improved usability for learners and educators.  
- Supported note types: **Whole, Half, Quarter, Eighth, Sixteenth**.  

### 🎼 Example Outputs  
- 📝 Handwritten Half Note → 🎵 Played Half Note  
- 📝 Handwritten Quarter Note → 🎵 Played Quarter Note  
- 📝 Handwritten Whole Note → 🎵 Played Whole Note  
- 📝 Handwritten Eighth Note → 🎵 Played Eighth Note  
- 📝 Handwritten Sixteenth Note → 🎵 Played Sixteenth Note
