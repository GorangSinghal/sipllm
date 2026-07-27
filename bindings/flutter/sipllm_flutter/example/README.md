# 🧠 Local LLM mobile app

A modern, high-performance Flutter application that serves as a unified mobile client for local Large Language Models. It allows seamless switching between **Llama.cpp**, **Ollama**, and **LM Studio** servers via local network or Ngrok tunnels.

## 📱 App UX

https://github.com/user-attachments/assets/fe605d22-2682-417e-aabd-f3841c0640ed

## ✨ Key Features

*   **Multi-Provider Support:** unified interface for multiple AI backends:
    *   🦙 **Llama.cpp** (via `llama-server`)
    *   🧶 **Ollama**
    *   🧪 **LM Studio**
*   **Modern UI:** Cyberpunk-inspired dark theme with neon accents and `lucide_icons`.
*   **Smart Connectivity:**
    *   Supports **Ngrok** tunneling for remote access.
    *   Supports **Localhost** (`10.0.2.2`) for Android Emulator.
    *   Auto-handling of API endpoints (`/v1/chat/completions`).
    *   Bypasses Ngrok browser warning pages via custom headers.
*   **Clean Architecture:** Feature-first structure with separated Logic (Providers), UI, and Services.

## 🛠 Tech Stack

*   **Framework:** Flutter & Dart
*   **State Management:** Provider (`ChangeNotifier`)
*   **Networking:** `http` (Direct API calls to control headers and timeouts)
*   **UI/UX:** `google_fonts` (Space Mono/JetBrains), `lucide_icons`

## 📂 Project Structure

The project follows a **Feature-First** architecture to ensure scalability.

```text
lib/
├── main.dart                   # Application Entry Point
├── core/                       # Core layer (Theme, Models)
│   ├── theme/                  # App Theme & Colors
│   └── models/                 # ChatMessage, LlmProvider Enum
├── services/                   # Network Layer
│   └── api_service.dart        # Universal HTTP Client for LLMs
└── features/                   # Feature layer
    ├── home/                   # Provider Selection Screen
    │   ├── home_screen.dart
    │   └── widgets/            # Provider Cards, Info Panels
    └── chat/                   # Chat Interface
        ├── chat_provider.dart  # State Management & Logic
        ├── chat_screen.dart
        └── widgets/            # Message Bubbles, Input Fields
```

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1️⃣ Prerequisites (Backend Setup)

Before running the app, ensure that at least one local LLM server is running.

**Option A — Llama.cpp**

Run the server and expose it to your local network:

`llama-server -m path/to/model.gguf --port 8081 --host 0.0.0.0`

**Option B — Ollama**

Important: Ollama must be configured to accept external requests.

`OLLAMA_HOST=0.0.0.0 OLLAMA_ORIGINS="*" ollama serve`

**Option C — LM Studio**

1. Open **LM Studio**
2. Go to **Developer Tab (<->)**
3. Start the server on **Port 1234**
4. Ensure **Cross-Origin Resource Sharing (CORS)** is enabled

### 2️⃣ Network Setup (Connecting Mobile to Localhost)

Since the app runs on a physical device or emulator, it cannot access `localhost` directly.

**Android Emulator**

Use the special Android host IP:

`http://10.0.2.2:PORT`

Example for Ollama:

`http://10.0.2.2:11434`

**Physical Devices (iOS / Android)**

Use **Ngrok** to expose your local server to the internet.

Example for LM Studio:

`ngrok http 1234`

Copy the generated `https://` URL and paste it into the app.

### 3️⃣ Running the App

Clone the repository:

`git clone https://github.com/YOUR-USERNAME/local-llm-hub.git`  
`cd local-llm-chat`

Install dependencies:

`flutter pub get`

Run the application:

`flutter run`

**iOS Note:**  
When running on a physical iPhone (iOS 17+), use release mode to avoid memory protection issues:

`flutter run --release`

## 📄 License

This project is provided for educational purposes as part of AI Frameworks Training.
