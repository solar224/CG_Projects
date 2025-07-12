# D3QN 強化學習賽車 AI

這是一個結合 C++/OpenGL 3D 賽車遊戲與 Python/PyTorch 深度強化學習的專案。AI Agent 採用了 **D3QN (Dueling Double Deep Q-Network)** 演算法，透過與遊戲環境的即時互動，從零開始學會自動駕駛並完成賽道。

![Project Demo](https://googleusercontent.com/image_generation_content/0)

## 主要功能

* **3D 賽車遊戲**：使用 C++ 和 OpenGL 從頭打造，包含完整的物理模擬、多種賽道表面（柏油、沙地、草地）和可互動的場景物件。
* **先進的 AI 演算法**：AI 的核心是 D3QN，它整合了 **Dueling DQN** 和 **Double DQN** 的優點，提升了學習的穩定性和效率。
* **即時通訊架構**：Python AI 代理與 C++ 遊戲環境之間透過 TCP Socket 進行低延遲的即時通訊，以交換狀態（State）和動作（Action）。
* **多種遊戲模式**：
    * **Solo**：單人計時賽。
    * **1 vs 1**：雙人對戰模式。
    * **1 vs PC**：玩家對抗訓練好的 AI。
* **完整的訓練流程**：提供詳細的訓練腳本 `train.py`，並支援：
    * **TensorBoard 視覺化**：即時監控獎勵、損失函數、Q值等關鍵指標。
    * **Email 通知系統**：在訓練完成、完成一圈或發生錯誤時自動發送郵件通知。
* **最佳紀錄重播系統**：在「1 vs PC」模式中，AI 會自動載入玩家在 Solo 模式中的最佳紀錄進行挑戰。

## 技術

* **遊戲端**：C++, OpenGL, GLUT
* **AI 端**：Python, PyTorch
* **函式庫**：`numpy`, `winsock2` (C++), `stb_image`, `tiny_obj_loader`

## 設定與安裝

請依照以下步驟設定您的本地開發環境。

### 1. 環境需求

* **C++ 編譯器**：建議在 Windows 上安裝 [MinGW-w64](https://www.mingw-w64.org/) 並將其 `bin` 目錄加入系統環境變數。
* **Python**：建議使用 Python 3.8 或更高版本。
* **GLUT 函式庫**：請確保您的編譯器可以找到 `glut32.lib`、`glu32.lib` 和 `opengl32.lib`。通常，這些是 Windows SDK 的一部分，或可與 MinGW 一同安裝。

### 2. 安裝步驟

1.  **clone 專案庫**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```

2.  **安裝 Python 依賴套件**
    ```bash
    pip install torch torchvision torchaudio numpy matplotlib
    ```

3.  **編譯 C++ 遊戲**
    在專案根目錄下打開終端機，執行以下指令來編譯遊戲。此指令會生成 `racing.exe`。
    ```bash
    g++ Final_Project.cpp -o racing.exe -I. -L. -lglut32 -lglu32 -lopengl32 -lwinmm -lws2_32 -limm32
    ```
    > **注意**：`-I.` 和 `-L.` 參數會告訴編譯器在當前目錄尋找標頭檔和函式庫。請確保 `glut.h` 等檔案位於正確位置。

## 使用說明

### 1. 訓練 AI 模型

執行 `train.py` 腳本來開始訓練。您可以自訂多種參數，一個功能完整的啟動指令如下：

```bash
python train.py --episodes 4000 --batch_size 256 --forward_pretraining --email_notifications --email_sender YOUR_EMAIL@gmail.com --email_recipient YOUR_EMAIL@gmail.com --email_password "xxxx xxxx xxxx xxxx" --notify_lap_completion --notify_training_completion --notify_errors
