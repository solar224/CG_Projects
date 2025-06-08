#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")
#include <mmsystem.h>
#include <windows.h>
#pragma comment(lib, "winmm.lib")
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include "glut.h"
#include "tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// 網路相關
bool initNetwork();
void checkForConnections();
void handleClientCommands();
void cleanupNetwork();
// 視窗/全螢幕 切換用
bool gFullScreen = false;
int gWinPosX = 50, gWinPosY = 50;        // 回到視窗模式時要擺回來的位置
int gWinWidth = 1640, gWinHeight = 720;  // 回到視窗模式時用的大小 (啟動時就是它)
bool previewCamMoveForward = false;
bool previewCamMoveBackward = false;
bool previewCamMoveLeft = false;
bool previewCamMoveRight = false;
GLuint texMenu = 0;
GLuint texSettlement = 0;  // 結算畫面背景紋理
GLuint texAsphalt = 0;     // 柏油路紋理 ID
GLuint texSand = 0;
GLuint texGrass = 0;
using namespace std;
HWND g_hWnd = NULL;
int winW = 1280, winH = 720;  // updated in reshape
enum GameMode { MODE_MENU, MODE_SOLO, MODE_PVP, MODE_AI, MODE_RESULT, MODE_LOADING, MODE_TRACK_PREVIEW };
GameMode gameMode = MODE_MENU;          // 初始顯示選單
GameMode previousGameMode = MODE_MENU;  // 儲存切換到結算畫面前的模式
GameMode targetGameMode = MODE_MENU;    // 儲存從載入畫面要跳轉的模式
float finalTimeP1 = 0.0f;               // 玩家1完成時間
float finalTimeP2 = 0.0f;               // 玩家2完成時間
int finalRankP1 = 0;                    // 玩家1排名
int finalRankP2 = 0;                    // 玩家2排名
bool p1Finished = false;                // 玩家1是否完成比賽
bool p2Finished = false;                // 玩家2是否完成比賽
int frameCount = 0;
int previousTime = 0;
float fps = 0.0f;
void drawFPS();
float loadingProgress = 0.0f;
float loadingTime = 0.0f;
const float LOADING_DURATION = 1.0f;
struct UIButton {
    int x, y, w, h;
    const char* text;
};
bool isSettingsMenuOpen = false;
UIButton btnSettingsIcon{15, 0, 40, 40, "S"};
UIButton btnReturnToMenuFromSettings{0, 0, 200, 50, "Return & Reset"};
// ---- 背景音樂相關 ----
bool backgroundMusicEnabled = true;                   // 預設開啟背景音樂
bool musicPlaying = false;                            // 目前音樂是否正在播放
UIButton btnToggleMusic{0, 0, 200, 50, "Music: ON"};  // 設定選單中的音樂開關按鈕

// ---- 音效相關 ----
bool sfxEnabled = true;                           // 預設開啟音效
bool isWindPlaying = false;                       // 風聲是否正在播放
UIButton btnToggleSFX{0, 0, 200, 50, "SFX: ON"};  // 設定選單中的音效開關按鈕
struct MaterialInfo {
    string name;
    float ambient[3] = {0.2f, 0.2f, 0.2f};
    float diffuse[3] = {0.8f, 0.8f, 0.8f};
    float specular[3] = {1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;
    GLuint texture_id = 0;  // 漫反射紋理 ID
};
struct MeshPart {
    vector<float> vertices;
    vector<float> normals;
    vector<float> texcoords;
    int material_index = -1;  // 對應到材質庫中的索引
};
struct ModelObject {
    vector<MeshPart> meshes;
    vector<MaterialInfo> materials;
};
ModelObject mustangModel, CasteliaCity, BusGameMap, Cliff, TropicalIslands, Cone1, Cone2;  // 為野馬模型新增一個物件/**
UIButton btnSolo{50, winH / 2 + 100, 180, 60, "Solo"};
UIButton btnPVP{50, winH / 2 + 20, 180, 60, "1 v 1"};
UIButton btnAI{50, winH / 2 - 60, 180, 60, "1 v PC"};
UIButton btnPreview{50, winH / 2 - 140, 180, 60, "Track Preview"};  // 新增賽道預覽按鈕
// Leaderboard
const char* SCORE_FILE = "scores.txt";
vector<float> scores;  // 依成績秒數遞增排序
const int TOP_N = 8;   // 只顯示前 8 名
const int DUST_NUM = 80;
const float CAM_H = 6.0f;                      // 基本高度
const float CAM_DIST = 8.0f;                   // 與車後距離
const float CAM_POS_SMOOTH_RATE = 5.0f;        // 位置平滑率：數值越大，跟隨越快 (反應更靈敏，平滑度降低)
const float CAM_LOOK_SMOOTH_RATE = 10.0f;      // 視角目標點平滑率
const float CAM_FOV_SMOOTH_RATE = 10.0f;       // FOV平滑率
const float MAX_DIST_LEASH_TOLERANCE = 10.0f;  // 拴繩容忍距離：攝影機實際位置可以比理想跟隨距離 (dynamicDist) 遠多少
const float LOOK_AHEAD = 6.0f;                 // 視線往前看多少
const float BASE_FOV = 60.0f;
const float MAX_FOV = 70.0f;      // 全速時的視角
const float MIN_CAM_DIST = 2.0f;  // 最小相機距離（高速時）
const float MAX_CAM_DIST = 8.0f;  // 最大相機距離（低速時）

void loadScores() {
    scores.clear();
    ifstream in(SCORE_FILE);
    float t;
    while (in >> t) scores.push_back(t);
    sort(scores.begin(), scores.end());
}
void saveScore(float t) {  // append + 重新排序
    ofstream out(SCORE_FILE, ios::app);
    out << fixed << setprecision(2) << t << '\n';
    out.close();
    loadScores();
}
struct Dust {
    float x, y;    // 2D 位置
    float r;       // 半徑
    float vx, vy;  // 速度 (px/s) >0 右上
    float a;       // alpha
};
vector<Dust> dust;
struct State {
    float x, y, z, heading, speed;
};
struct Cloud {
    float x, y;   // 左下角座標
    float w, h;   // 雲寬高
    float speed;  // px/sec
};
vector<Cloud> clouds;
struct Vec3 {
    float x{}, y{}, z{};
    Vec3() = default;
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    Vec3 operator+(const Vec3& o) const {
        return {x + o.x, y + o.y, z + o.z};
    }
    Vec3 operator-(const Vec3& o) const {
        return {x - o.x, y - o.y, z - o.z};
    }
    Vec3 operator*(float s) const {
        return {x * s, y * s, z * s};
    }
};
// 滑鼠拖曳相關變數
bool isMouseDragging = false;
int lastMouseX = 0, lastMouseY = 0;
Vec3 originalCamPos, originalLookTarget;    // 保存原始攝像機位置
Vec3 originalCam2Pos, originalLookTarget2;  // 第二台車的攝像機
float originalFov, originalFov2;            // 原始視野
bool startCameraRestoration = false;        // 控制攝影機恢復動畫
/* ============  Camera  ============ */
struct Camera {
    Vec3 pos;                              // 畫面所在位置
    float fov = 60.0f;                     // 目前 FOV
    Vec3 lookTarget;                       // 新增：儲存平滑化後的目標點
} cam, cam2;                               // Add a second camera for car2
const float CAR_COLLISION_RADIUS = 4.5f;   // 車輛碰撞半徑 (可根據模型大小微調)
const float COLLISION_RESTITUTION = 0.8f;  // 碰撞恢復係數 (0.0 = 完全無彈性, 1.0 = 完全彈性)
void handleCarCollision();
static float length(const Vec3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
static Vec3 normalize(const Vec3& v) {
    float l = length(v);
    if (l < 1e-6f) return {0, 0, 0};
    return {v.x / l, v.y / l, v.z / l};
}
// 外積函式
static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// 記錄上一幀車輛是否在終點區域內，以檢測“進入”
static bool car1WasInFinishVolume = false;
static bool car2WasInFinishVolume = false;
static Vec3 catmull(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return (p1 * 2.0f + (p2 - p0) * t + (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t2 +
            (p3 - p0 + (p1 - p2) * 3.0f) * t3) *
           0.5f;
}
vector<Vec3> ctrl = {{360, 0, 0},     {360, 0, 200},  {240, 0, 360}, {120, 0, 450},   {-150, 0, 430},
                     {-280, 0, 300},  {-330, 0, 120}, {-400, 0, 0},  {-360, 0, -120}, {-240, 0, -240},
                     {-160, 0, -360}, {-40, 0, -400}, {80, 0, -320}, {120, 0, -200}};
vector<Vec3> trackCtrl;
vector<Vec3> trackSamples;
// ---- 起點 / 終點拱門位置與朝向 ----
Vec3 gateStartPos, gateFinishPos;
float gateStartHdg = 0.0f, gateFinishHdg = 0.0f;
constexpr int SAMPLE_DENSITY = 2000;
float trackLength = 0.0f;
const float ROAD_HALF_W = 20.0f;
// ----- 路面分區 -----
const float SAND_HALF_W = 60.0f;  // 沙地外緣（中心線 ±20u）
const float MAX_F_ROAD = 120.0f;  // 柏油最高速
const float MAX_F_SAND = 70.0f;   // 沙地最高速
const float MAX_F_GRASS = 25.0f;  // 草地最高速（失控）
// Mini‑map extents
float mapMinX, mapMaxX, mapMinZ, mapMaxZ;
int prevMs = 0;
// --- 即時排名距離量測 ---
float travelP1 = 0.0f;     // 賽道進度 (car)
float travelP2 = 0.0f;     // 賽道進度 (car2)
int lastCheckpointP1 = 0;  // 車輛1最後經過的賽道檢查點索引
int lastCheckpointP2 = 0;  // 車輛2最後經過的賽道檢查點索引
int lapP1 = 0;             // 車輛1的圈數
int lapP2 = 0;             // 車輛2的圈數
bool countdownActive = false;
float countdownLeft = 0.0f;                 // 秒
bool forceSettlementActive = false;         // 強制結算倒計時標誌
float forceSettlementTimer = 0.0f;          // 強制結算倒計時(秒)
const float FORCE_SETTLEMENT_TIME = 10.0f;  // 強制結算等待時間(秒)
// ---- 終點偵測 ----
Vec3 finishVolumeMin;                                  // 終點空間的最小角點 (min x, min y, min z)
Vec3 finishVolumeMax;                                  // 終點空間的最大角點 (max x, max y, max z)
const float FINISH_VOLUME_WIDTH = ROAD_HALF_W * 5.0f;  // 終點空間的寬度 (略大於路寬)
const float FINISH_VOLUME_HEIGHT = 10.0f;              // 終點空間的高度 (確保能覆蓋車輛)
const float FINISH_VOLUME_DEPTH = 5.0f;                // 終點空間的深度 (沿賽道方向的厚度)
// ---- 賽車計時器 ----
float raceTime = 0.0f;  // 累積秒數
bool timerRunning = true;
bool firstFrame = true;  // 用來取得第一幀的側向基準
const int UI_BTN_W0 = 180;
const int UI_BTN_H0 = 60;
const int UI_BTN_X0 = 50;
const int UI_BTN_GAP = 80;   // 垂直間距 (PVP ↔ AI)
const int UI_TITLE_OF = 40;  // 標題距上方按鈕的位移
void drawPlayerLabel(int playerNum, bool isLeft);
void drawVerticalDivider();
void drawSpeedometer(float speed, bool isLeftSide = false);
void drawLiveRanking(bool isLeftSide = false);
void drawForceSettlementTimer(bool isLeftSide = false);  // 添加強制結算倒計時顯示函數聲明
void drawHUDTimer(bool isLeftSide = false);              // 更新計時器函數聲明

void drawRail(const Vec3& start, const Vec3& end, float beam_height, float beam_depth);  // 繪製護欄組件
void drawTrackBarriers();                                                                // 繪製賽道圍欄

void drawCurbstones();     // 繪製紅白路緣石
void drawLoadingScreen();  // 繪製載入畫面
void drawFPS();            // 繪製FPS的函式
void drawEnvironmentModels();
void drawTrackPreviewUI();
void drawFinishVolume();  // 繪製終點檢測空間
void fullResetToMenu();
void drawSettingsElements(int, int);  // 繪製設定按鈕和彈出選單
void drawBackground();
void drawTrafficCones();

void setImeMode(bool active) {
    if (g_hWnd == NULL) {
        // 第一次呼叫時，找到目前的視窗並儲存其控制代碼
        g_hWnd = FindWindow(NULL, "Speed Racing");
        if (g_hWnd == NULL) {
            cerr << "[ERROR] Cannot find window to set IME mode." << endl;
            return;
        }
    }

    // 取得輸入法上下文 (Input Method Context)
    HIMC hImc = ImmGetContext(g_hWnd);
    if (hImc) {
        // 開啟或關閉輸入法
        ImmSetOpenStatus(hImc, active);
        // 釋放上下文
        ImmReleaseContext(g_hWnd, hImc);
    }
}
// ---- 背景音樂控制函式 ----
void playBackgroundMusic(const char* filePath) {
    if (!backgroundMusicEnabled || musicPlaying) return;
    mciSendString("close bgmusic", NULL, 0, NULL);
    string openCmd = "open \"" + string(filePath) + "\" alias bgmusic";
    if (mciSendString(openCmd.c_str(), NULL, 0, NULL) == 0) {
        mciSendString("setaudio bgmusic volume to 100", NULL, 0, NULL);

        if (mciSendString("play bgmusic repeat", NULL, 0, NULL) == 0) {
            musicPlaying = true;
            cout << "[INFO] Background music started: " << filePath << endl;
        } else {
            cerr << "[ERROR] Failed to play background music." << endl;
            mciSendString("close bgmusic", NULL, 0, NULL);
        }
    } else {
        cerr << "[ERROR] Failed to open background music file: " << filePath << endl;
    }
}

void stopBackgroundMusic() {
    if (musicPlaying) {
        mciSendString("stop bgmusic", NULL, 0, NULL);
        mciSendString("close bgmusic", NULL, 0, NULL);
        musicPlaying = false;
        cout << "[INFO] Background music stopped." << endl;
    }
    mciSendString("close sfx", NULL, 0, NULL);
}
// ---- 播放一次性音效的函數 ----
// 輔助函數：印出詳細的 MCI 錯誤訊息
void printMCIError(DWORD_PTR dwError) {
    char errorText[256];
    if (mciGetErrorString(dwError, errorText, sizeof(errorText))) {
        // 如果能成功取得錯誤字串
        cerr << "[MCI ERROR] Code " << dwError << ": " << errorText << endl;
    } else {
        // 如果連錯誤字串都拿不到
        cerr << "[MCI ERROR] Code " << dwError << ": An unknown error occurred." << endl;
    }
}
// ---- 播放循環音效的通用函數 ----
void playLoopingSound(const char* alias, const char* filePath) {
    string openCmd = "open \"" + string(filePath) + "\" type mpegvideo alias " + string(alias);
    if (mciSendString(openCmd.c_str(), NULL, 0, NULL) == 0) {
        string playCmd = "play " + string(alias) + " repeat";
        mciSendString(playCmd.c_str(), NULL, 0, NULL);
    } else {
        cerr << "[ERROR] Failed to open looping sound file: " << filePath << endl;
    }
}

// ---- 停止循環音效的通用函數 ----
void stopLoopingSound(const char* alias) {
    string closeCmd = "close " + string(alias);
    mciSendString(closeCmd.c_str(), NULL, 0, NULL);
}
void playSoundEffect(const char* filePath) {
    if (!sfxEnabled) {
        return;
    }

    // 之前會關閉其他同名音效，保留這個行為
    mciSendString("close click", NULL, 0, NULL);

    // *** 核心修改：移除 "type waveaudio"，讓MCI自動判斷 ***
    // 並且使用一個專門的別名 "click" 來播放按鈕音效，避免與其他音效衝突
    string openCmd = "open \"" + string(filePath) + "\" alias click";

    DWORD_PTR dwError = mciSendString(openCmd.c_str(), NULL, 0, NULL);
    if (dwError == 0) {
        // 從頭開始播放
        mciSendString("play click from 0", NULL, 0, NULL);
    } else {
        cerr << "[ERROR] Failed to open sound effect file: " << filePath << endl;
        printMCIError(dwError);
    }
}

// ---- 背景音樂控制函式 ----
void toggleBackgroundMusic() {
    backgroundMusicEnabled = !backgroundMusicEnabled;
    btnToggleMusic.text = backgroundMusicEnabled ? "Music: ON" : "Music: OFF";
    if (backgroundMusicEnabled) {
        if (gameMode == MODE_MENU && !musicPlaying) {
            playBackgroundMusic("Menu.mp3");
        } else if ((gameMode == MODE_SOLO || gameMode == MODE_PVP || gameMode == MODE_AI) && !musicPlaying) {
            playBackgroundMusic("game.mp3");
        } else if (gameMode == MODE_RESULT && !musicPlaying) {
            playBackgroundMusic("Result.mp3");
        }
    } else {
        stopBackgroundMusic();
    }
    std::cout << "[INFO] Background music " << (backgroundMusicEnabled ? "enabled" : "disabled") << std::endl;
}
// ---- 音效(SFX)控制函式 ----
void toggleSfx() {
    sfxEnabled = !sfxEnabled;
    btnToggleSFX.text = sfxEnabled ? "SFX: ON" : "SFX: OFF";
    if (!sfxEnabled) {
        // 如果關閉音效，立刻停止風聲
        if (isWindPlaying) {
            stopLoopingSound("wind");
            isWindPlaying = false;
        }
    }
    std::cout << "[INFO] Sound effects " << (sfxEnabled ? "enabled" : "disabled") << std::endl;
}

float lateralOffsetFromCenter(const Vec3& pos) {
    float best2 = FLT_MAX;
    for (const auto& p : trackSamples) {
        Vec3 d = pos - p;
        float d2 = d.x * d.x + d.z * d.z;
        if (d2 < best2) best2 = d2;
    }
    return sqrtf(best2);
}
struct PlayerInputAction {
    float timestamp;    // 從比賽開始的時間戳 (秒)
    unsigned char key;  // 按下的鍵 (例如 'W', 'A', 'S', 'D', 或特殊鍵的代碼)
    bool isDown;        // true 表示按下, false 表示釋放
};
class Car {
   public:
    float x = 0, y = 0, z = 0, speed = 0, heading = 0;
    vector<State> log;
    bool kW = false, kS = false, kA = false, kD = false;
    float current_steer_force = 0.0f;      // 目前的轉向力度 (-1.0 到 1.0)
    const float STEER_CHANGE_RATE = 2.5f;  // 方向盤轉動/回正速率
    // 較小的值會使轉向和回正更慢、更平滑
    // 較大的值會使轉向和回正更快、更靈敏
    bool isAIControlledByReplay = false;       // 新增：標記是否由 Solo 記錄控制
    vector<State> replayLogToFollow;           // 新增：儲存要播放的 Solo 記錄
    int currentReplayFrameIndex = 0;           // 新增：目前播放到 Solo 記錄的哪一幀
    vector<PlayerInputAction> recordedInputs;  // 儲存玩家的按鍵操作序列
    bool isAIControlledByInputReplay = false;  // 新標記，表示由按鍵重播控制
    int currentReplayInputActionIndex = 0;     // 目前播放到哪個按鍵操作

    void update(float dt) {
        if (isAIControlledByInputReplay) {
            // 檢查是否有下一個按鍵操作，並且時間戳已到達或超過
            while (currentReplayInputActionIndex < recordedInputs.size() &&
                   raceTime >= recordedInputs[currentReplayInputActionIndex].timestamp) {
                const PlayerInputAction& action = recordedInputs[currentReplayInputActionIndex];
                char keyChar = action.key;  // 記錄的是大寫鍵
                if (keyChar == 'W')
                    kW = action.isDown;
                else if (keyChar == 'S')
                    kS = action.isDown;
                else if (keyChar == 'A')
                    kA = action.isDown;
                else if (keyChar == 'D')
                    kD = action.isDown;
                currentReplayInputActionIndex++;
            }
        } else if (isAIControlledByReplay) {
            if (currentReplayFrameIndex < replayLogToFollow.size()) {
                const State& targetState = replayLogToFollow[currentReplayFrameIndex];
                this->x = targetState.x;
                this->y = targetState.y;
                this->z = targetState.z;
                this->heading = targetState.heading;
                this->speed = targetState.speed;
                currentReplayFrameIndex++;
            } else {
                this->speed *= 0.98f;
            }
            log.push_back({x, y, z, heading, speed});
            return;
        }
        float lateral = lateralOffsetFromCenter({x, y, z});
        bool onRoad = (lateral <= ROAD_HALF_W);
        bool onSand = (lateral > ROAD_HALF_W && lateral <= SAND_HALF_W);
        float localMaxF = onRoad ? MAX_F_ROAD : onSand ? MAX_F_SAND : MAX_F_GRASS;
        const float ACC_BASE = 12.0f;
        const float ACC = ACC_BASE * (localMaxF / MAX_F_ROAD);
        const float BRAKE = 24.0f * (localMaxF / MAX_F_ROAD);
        const float TURN = 90.0f * (localMaxF / MAX_F_ROAD);
        float friction = onRoad ? 0.998f : onSand ? 0.99f : 0.90f;
        /*------------------------------------------------------------
         1) 引擎 / 煞車 / 自然滑行
     ------------------------------------------------------------*/
        if (kW)
            speed += ACC * dt;
        else if (kS)
            speed -= BRAKE * dt;
        else
            speed *= friction;
        /*------------------------------------------------------------
            2) 平滑處理轉向輸入並更新 current_steer_force
        ------------------------------------------------------------*/
        float target_steer_force = 0.0f;
        if (kA) target_steer_force = 1.0f;   // 左轉
        if (kD) target_steer_force = -1.0f;  // 右轉 (如果kA和kD同時按下，它們會互相抵消或取決於最後一個if)
        if (current_steer_force < target_steer_force) {
            current_steer_force += STEER_CHANGE_RATE * dt;
            if (current_steer_force > target_steer_force) current_steer_force = target_steer_force;

        } else if (current_steer_force > target_steer_force) {
            current_steer_force -= STEER_CHANGE_RATE * dt;
            if (current_steer_force < target_steer_force) current_steer_force = target_steer_force;
        }
        current_steer_force = max(-1.0f, min(1.0f, current_steer_force));
        /*------------------------------------------------------------
            2.5) 轉向拖速 (使用平滑後的 current_steer_force)
        ------------------------------------------------------------*/
        const float TURN_DRAG_K = 1.2f;
        float vNorm = fabsf(speed) / localMaxF;  // 0 ~ 1
        speed -= fabsf(current_steer_force) * TURN_DRAG_K * vNorm * vNorm * localMaxF * dt;
        /*------------------------------------------------------------
            3) 更新方向角與位置 (使用平滑後的 current_steer_force)
        ------------------------------------------------------------*/
        heading += current_steer_force * TURN * dt * (speed / localMaxF);
        float r = heading * M_PI / 180;
        x += sinf(r) * speed * dt;
        z += cosf(r) * speed * dt;
        /*------------------------------------------------------------
            4) 速度邊界
        ------------------------------------------------------------*/
        const float MAX_R = -20.0f;
        speed = max(MAX_R, min(speed, localMaxF));
        log.push_back({x, y, z, heading, speed});
    }
    void saveLog(const string& fn = "state_log.csv") {
        ofstream o(fn);
        o << "x,y,z,heading,speed\n";
        for (auto& s : log) o << s.x << "," << s.y << "," << s.z << "," << s.heading << "," << s.speed << "\n";
        cerr << "Saved " << log.size() << " frames to " << fn << "\n";
    }
} car, car2;
struct SoloReplayInfo {
    float time;          // 完成時間
    string logFilePath;  // 對應的 .csv 檔案路徑
    bool operator<(const SoloReplayInfo& other) const {
        return time < other.time;  // 時間越少越好
    }
};
vector<SoloReplayInfo> topSoloReplays;
const char* REPLAY_INFO_FILE = "solo_replays.txt";  // 儲存前三名記錄資訊的檔案
const int MAX_TOP_REPLAYS = 3;                      // 只保留前三名

static const GLfloat PLANE[4] = {0.f, 1.f, 0.f, -0.02f};
static const GLfloat LIGHT[4] = {0.35f, 1.f, 0.25f, 0.0f};  // w=0
static GLfloat SHADOW[16];
// ---- 更新引擎/環境音效 ----
void updateEngineSound() {
    // 如果音效被禁用，則確保風聲是停止的並直接返回
    if (!sfxEnabled) {
        if (isWindPlaying) {
            stopLoopingSound("wind");
            isWindPlaying = false;
        }
        return;
    }

    // 當車速超過一個閾值時播放風聲
    const float WIND_SPEED_THRESHOLD = 2.0f;

    // *** 核心修改：判斷任一車輛是否有速度 ***
    bool anyCarMoving = false;
    if (gameMode == MODE_SOLO) {
        // 單人模式下，只判斷 car 1
        anyCarMoving = (car.speed > WIND_SPEED_THRESHOLD);
    } else if (gameMode == MODE_PVP || gameMode == MODE_AI) {
        // 多人模式下，判斷 car 1 或 car 2
        anyCarMoving = (car.speed > WIND_SPEED_THRESHOLD || car2.speed > WIND_SPEED_THRESHOLD);
    }

    if (anyCarMoving) {
        if (!isWindPlaying) {
            playLoopingSound("wind", "windNoise.mp3");
            isWindPlaying = true;
        }
    } else {
        if (isWindPlaying) {
            stopLoopingSound("wind");
            isWindPlaying = false;
        }
    }
}

// 載入 Solo 操作記錄資訊
void loadTopSoloReplays() {
    topSoloReplays.clear();
    ifstream inFile(REPLAY_INFO_FILE);
    if (!inFile) {
        cerr << "[INFO] No solo replay info file found (" << REPLAY_INFO_FILE << "). Starting fresh." << endl;
        return;
    }
    SoloReplayInfo tempInfo;
    while (inFile >> tempInfo.time >> tempInfo.logFilePath) topSoloReplays.push_back(tempInfo);
    inFile.close();
    sort(topSoloReplays.begin(), topSoloReplays.end());  // 確保按時間排序
    cout << "[INFO] Loaded " << topSoloReplays.size() << " solo replay entries." << endl;
}

// 儲存 Solo 操作記錄資訊到檔案
void saveTopSoloReplaysToFile() {
    ofstream outFile(REPLAY_INFO_FILE);
    if (!outFile) {
        cerr << "[ERROR] Could not open solo replay info file for writing: " << REPLAY_INFO_FILE << endl;
        return;
    }
    for (const auto& replay : topSoloReplays)
        outFile << fixed << setprecision(2) << replay.time << " " << replay.logFilePath << endl;
    outFile.close();
    cout << "[INFO] Saved " << topSoloReplays.size() << " solo replay entries to " << REPLAY_INFO_FILE << endl;
}

// 當 Solo 模式完成時，儲存操作記錄並更新排行榜
void saveSoloOperationAndManageReplays(float newTime) {
    if (gameMode != MODE_SOLO) return;
    time_t now_time = time(nullptr);
    tm* ltm = localtime(&now_time);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "solo_input_log_%Y%m%d_%H%M%S.inputrec", ltm);
    string newLogFilename(buffer);
    ofstream outFile(newLogFilename);
    if (!outFile) {
        cerr << "[ERROR] Could not open input replay file for writing: " << newLogFilename << endl;
        return;
    }
    outFile << "timestamp,key,is_down" << endl;
    for (const auto& inputAction : car.recordedInputs)
        outFile << fixed << setprecision(4) << inputAction.timestamp << "," << inputAction.key << ","
                << (inputAction.isDown ? 1 : 0) << endl;
    outFile.close();
    cout << "[INFO] Saved " << car.recordedInputs.size() << " input actions to " << newLogFilename << endl;
    topSoloReplays.push_back({newTime, newLogFilename});
    sort(topSoloReplays.begin(), topSoloReplays.end());
    while (topSoloReplays.size() > MAX_TOP_REPLAYS) {
        SoloReplayInfo replayToRemove = topSoloReplays.back();
        if (remove(replayToRemove.logFilePath.c_str()) != 0)
            cerr << "[ERROR] Failed to delete old input replay file: " << replayToRemove.logFilePath << endl;
        else
            cout << "[INFO] Deleted old input replay file: " << replayToRemove.logFilePath << endl;

        topSoloReplays.pop_back();
    }
    saveTopSoloReplaysToFile();
}

bool loadInputReplayLogFromFile(const string& filePath, vector<PlayerInputAction>& inputLog) {
    inputLog.clear();
    ifstream inFile(filePath);
    if (!inFile) {
        cerr << "[ERROR] Could not open input replay log file: " << filePath << endl;
        return false;
    }
    string line;
    if (!getline(inFile, line)) {
        cerr << "[ERROR] Input replay log file is empty or cannot read header: " << filePath << endl;
        inFile.close();
        return false;
    }
    PlayerInputAction tempAction;
    int isDownInt;
    int lineNum = 1;  // 從數據的第1行開始算 (標題行之後)
    while (getline(inFile, line)) {
        lineNum++;
        stringstream ss(line);
        char keyChar;  // 用於讀取單個按鍵字元
        char comma1, comma2;
        if (ss >> tempAction.timestamp &&                          // 1. 讀取時間戳
            ss >> comma1 && comma1 == ',' &&                       // 2. 讀取並驗證第一個逗號
            ss >> keyChar &&                                       // 3. 讀取按鍵字元
            ss >> comma2 && comma2 == ',' &&                       // 4. 讀取並驗證第二個逗號
            ss >> isDownInt) {                                     // 5. 讀取 isDown (0 或 1)
            tempAction.key = static_cast<unsigned char>(keyChar);  // 賦值給 PlayerInputAction 的 key
            tempAction.isDown = (isDownInt == 1);
            inputLog.push_back(tempAction);
        } else {
            cerr << "[WARNING] Failed to parse line " << lineNum << " in input replay log: \"" << line << "\"" << endl;
            if (ss.fail())
                cerr << "           Stream state (fail): badbit=" << ss.bad() << ", failbit=" << ss.fail()
                     << ", eofbit=" << ss.eof() << endl;
            else if (ss.bad())
                cerr << "           Stream state (bad): badbit=" << ss.bad() << ", failbit=" << ss.fail()
                     << ", eofbit=" << ss.eof() << endl;
            ss.clear();
        }
    }
    inFile.close();
    cout << "[INFO] Loaded " << inputLog.size() << " input actions from replay: " << filePath << endl;
    return !inputLog.empty();
}
bool loadReplayLogFromFile(const string& filePath, vector<State>& replayLog) {
    replayLog.clear();
    ifstream inFile(filePath);
    if (!inFile) {
        cerr << "[ERROR] Could not open replay log file: " << filePath << endl;
        return false;
    }
    string line;
    getline(inFile, line);  // 跳過標題行 "x,y,z,heading,speed"
    State tempState;
    char comma;  // 用於讀取逗號
    while (getline(inFile, line)) {
        stringstream ss(line);
        if (ss >> tempState.x >> comma >> tempState.y >> comma >> tempState.z >> comma >> tempState.heading >> comma >>
            tempState.speed) {
            replayLog.push_back(tempState);
        } else {
            cerr << "[WARNING] Failed to parse line in replay log: " << line << endl;
        }
    }
    inFile.close();
    cout << "[INFO] Loaded " << replayLog.size() << " frames from replay log: " << filePath << endl;
    return !replayLog.empty();
}
// ---- 三角錐相關 ----
struct TrafficCone {
    Vec3 pos;              // 位置
    Vec3 velocity;         // 速度 (被撞飛時)
    Vec3 angularVelocity;  // 角速度 (被撞飛時旋轉)
    float rotationAngle;   // 當前旋轉角度
    Vec3 rotationAxis;     // 旋轉軸
    bool isHit;            // 是否被撞飛
    bool isActive;         // 是否在場景中活動 (可用於重置或移除)
    float scale;           // 大小比例
    ModelObject* model;    // 指向要使用的模型 (例如 Cone1 或 Cone2)

    TrafficCone() : isHit(false), isActive(true), rotationAngle(0.0f), scale(1.0f), model(nullptr) {
        rotationAxis = {0.0f, 1.0f, 0.0f};  // 預設繞Y軸旋轉
    }
};

vector<TrafficCone> trafficCones;
const float CONE_COLLISION_RADIUS_CAR = 1.5f;                // 車輛與三角錐的碰撞半徑
const float CONE_COLLISION_RADIUS_CONE = 0.5f;               // 三角錐本身的碰撞半徑 (用於視覺)
const float CONE_PLACEMENT_OFFSET_MIN = ROAD_HALF_W + 2.0f;  // 離賽道中心線最小距離
const float CONE_PLACEMENT_OFFSET_MAX = SAND_HALF_W - 2.0f;  // 離賽道中心線最大距離 (確保在沙地上)
const float CONE_HIT_IMPULSE_MIN = 20.0f;
const float CONE_HIT_IMPULSE_MAX = 40.0f;
const float CONE_GRAVITY = -18.0f;
const float CONE_FRICTION = 0.98f;
const float CONE_ANGULAR_FRICTION = 0.97f;
const float CONE_MIN_VELOCITY_SQR = 0.1f * 0.1f;          // 速度平方小於此值則停止
const float CONE_MIN_ANGULAR_VELOCITY_SQR = 0.5f * 0.5f;  // 角速度平方小於此值則停止旋轉
void initializeTrafficCones();
void updateTrafficCones(float dt);
void handleCarConeCollision(Car& carToCollide, float dt);
// 穿越終點偵測
bool canFinishP1 = (lastCheckpointP1 > trackSamples.size() * 0.50f);
bool canFinishP2 = (lastCheckpointP2 > trackSamples.size() * 0.50f);
const int LAPS_TO_FINISH = 1;  // 設定比賽需要完成的圈數
int nearestSampleIndex(const Vec3& pos) {
    int bestIdx = 0;
    float best2 = FLT_MAX;
    for (int i = 0; i < (int)trackSamples.size(); ++i) {
        Vec3 d = pos - trackSamples[i];
        float d2 = d.x * d.x + d.z * d.z;
        if (d2 < best2) {
            best2 = d2;
            bestIdx = i;
        }
    }
    return bestIdx;
}
bool loadMultiMaterialModel(ModelObject& model, const string& modelPath) {
    tinyobj::attrib_t attrib;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;
    string warn, err;
    string mtl_basedir = modelPath.substr(0, modelPath.find_last_of("/\\") + 1);
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str(), mtl_basedir.c_str())) {
        cerr << "[ERROR] Failed to load model: " << modelPath << endl;
        if (!warn.empty()) cerr << "[WARN] " << warn << endl;
        if (!err.empty()) cerr << "[ERROR] " << err << endl;
        return false;
    }
    if (!warn.empty()) cout << "[WARN] tinyobj: " << warn << endl;  // 顯示警告
    cout << "[INFO] tinyobj::LoadObj finished. Shapes: " << shapes.size() << ", Materials: " << materials.size()
         << endl;
    // --- 1. 預先載入所有材質和紋理 ---
    map<string, GLuint> texture_cache;  // 避免重複載入相同紋理
    model.materials.reserve(materials.size());
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& mat = materials[i];
        cout << "[INFO] Material " << (i + 1) << "/" << materials.size() << ": " << mat.name;
        MaterialInfo material_info;
        material_info.name = mat.name;
        memcpy(material_info.ambient, mat.ambient, sizeof(float) * 3);
        memcpy(material_info.diffuse, mat.diffuse, sizeof(float) * 3);
        memcpy(material_info.specular, mat.specular, sizeof(float) * 3);
        material_info.shininess = mat.shininess;
        if (!mat.diffuse_texname.empty()) {
            string texture_filename_from_mtl = mat.diffuse_texname;
            replace(texture_filename_from_mtl.begin(), texture_filename_from_mtl.end(), '\\', '/');
            string texture_path = mtl_basedir + texture_filename_from_mtl;
            cout << " -> Tex: " << texture_filename_from_mtl;
            // 檢查快取
            if (texture_cache.find(texture_path) != texture_cache.end()) {
                material_info.texture_id = texture_cache[texture_path];
                cout << " (cached)" << endl;
            } else {
                cout << " (loading from: " << texture_path << ")" << endl;
                int w, h, n;
                unsigned char* data = stbi_load(texture_path.c_str(), &w, &h, &n, 0);
                if (data) {
                    GLuint tex_id;
                    glGenTextures(1, &tex_id);
                    glBindTexture(GL_TEXTURE_2D, tex_id);
                    GLenum format = (n == 3) ? GL_RGB : GL_RGBA;
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, format, GL_UNSIGNED_BYTE, data);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, w, h, format, GL_UNSIGNED_BYTE,
                                      data);  // 使用 GLU 建立 Mipmap
                    stbi_image_free(data);
                    material_info.texture_id = tex_id;
                    texture_cache[texture_path] = tex_id;  // 存入快取
                } else {
                    cerr << "[ERROR] Failed to load texture: " << texture_path << " reason: " << stbi_failure_reason()
                         << endl;
                }
            }
        } else {
            cout << " (no diffuse texture)" << endl;  // 如果沒有漫反射紋理則換行
        }
        model.materials.push_back(material_info);
    }
    cout << "[INFO] All materials processed." << endl;

    // --- 2. 根據材質ID，將頂點分組到不同的 MeshPart ---
    for (size_t s = 0; s < shapes.size(); ++s) {  // -1 表示還沒有分配材質
        const auto& shape = shapes[s];
        int current_material_id = -1;

        for (size_t i = 0; i < shape.mesh.indices.size() / 3; i++) {
            int face_material_id = shape.mesh.material_ids[i];

            // 如果這個面的材質和上一個不同，就需要建立或切換到對應的 MeshPart
            if (face_material_id != current_material_id) {
                current_material_id = face_material_id;
                // 尋找是否已經有對應該材質的 MeshPart
                auto it = find_if(model.meshes.begin(), model.meshes.end(),
                                  [&](const MeshPart& m) { return m.material_index == current_material_id; });

                if (it == model.meshes.end()) {
                    // 沒找到，建立一個新的 MeshPart
                    MeshPart new_mesh;
                    new_mesh.material_index = current_material_id;
                    model.meshes.push_back(new_mesh);
                }
            }
            MeshPart& current_mesh = model.meshes.back();
            // 處理這個面的三個頂點
            for (int j = 0; j < 3; j++) {
                tinyobj::index_t idx = shape.mesh.indices[3 * i + j];
                current_mesh.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
                current_mesh.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
                current_mesh.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 2]);
                if (idx.normal_index >= 0) {
                    current_mesh.normals.push_back(attrib.normals[3 * idx.normal_index + 0]);
                    current_mesh.normals.push_back(attrib.normals[3 * idx.normal_index + 1]);
                    current_mesh.normals.push_back(attrib.normals[3 * idx.normal_index + 2]);
                }
                if (idx.texcoord_index >= 0) {
                    current_mesh.texcoords.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
                    current_mesh.texcoords.push_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
                }
            }
        }
    }
    cout << "[INFO] All shapes processed. Model has " << model.meshes.size() << " mesh parts." << endl;
    return true;
}

// ---- 繪製FPS的函式 ----
void drawFPS() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);  // 左下 (0,0)
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    char fpsText[32];
    sprintf(fpsText, "FPS: %.1f", fps);
    glColor3f(1.0f, 1.0f, 1.0f);
    // 設定文字位置在右上角
    int textWidth = 0;
    for (char* p = fpsText; *p; ++p) textWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_18, *p);

    int x = winW - textWidth - 10;  // 10 pixels from the right edge
    int y = winH - 20;              // 20 pixels from the top edge
    glRasterPos2i(x, y);
    for (char* p = fpsText; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ---- 能繪製多材質模型的函式 ----
void drawMultiMaterialModel(const ModelObject& model) {
    glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT);
    glEnable(GL_COLOR_MATERIAL);

    for (const auto& mesh : model.meshes) {
        if (mesh.vertices.empty()) continue;

        // --- 1. 設定材質 ---
        if (mesh.material_index >= 0 && mesh.material_index < model.materials.size()) {
            const MaterialInfo& mat = model.materials[mesh.material_index];

            // 設定顏色和光照屬性
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat.ambient);
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat.diffuse);
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat.specular);
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, mat.shininess);

            // 綁定紋理
            if (mat.texture_id > 0) {
                glEnable(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D, mat.texture_id);
                // 讓紋理顏色與光照顏色混合
                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
            } else {
                glDisable(GL_TEXTURE_2D);
            }
        } else {
            // 沒有材質，使用預設值
            glDisable(GL_TEXTURE_2D);
        }

        // --- 2. 傳遞頂點數據並繪製 ---
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, mesh.vertices.data());

        if (!mesh.normals.empty()) {
            glEnableClientState(GL_NORMAL_ARRAY);
            glNormalPointer(GL_FLOAT, 0, mesh.normals.data());
        }
        if (!mesh.texcoords.empty()) {
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);
            glTexCoordPointer(2, GL_FLOAT, 0, mesh.texcoords.data());
        }

        glDrawArrays(GL_TRIANGLES, 0, mesh.vertices.size() / 3);

        // 清理 Client State
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    glPopAttrib();  // 還原 OpenGL 狀態
}

void buildTrack() {
    vector<Vec3> tmp = ctrl;
    tmp.push_back(ctrl[0]);
    tmp.push_back(ctrl[1]);
    int SEGS = (int)tmp.size() - 3;
    vector<Vec3> raw;
    for (int s = 0; s < SEGS; ++s) {
        for (int i = 0; i < SAMPLE_DENSITY / SEGS; ++i) {
            float t = (float)i / (SAMPLE_DENSITY / SEGS);
            raw.push_back(catmull(tmp[s], tmp[s + 1], tmp[s + 2], tmp[s + 3], t));
        }
    }
    float len = 0.0f;
    for (size_t i = 1; i < raw.size(); ++i) len += length(raw[i] - raw[i - 1]);
    const float TARGET_LEN = 8100.0f;
    float scale = TARGET_LEN / len;
    trackCtrl.clear();
    trackCtrl.reserve(ctrl.size());
    for (auto& p : ctrl) trackCtrl.push_back({p.x * scale, p.y, p.z * scale});
    vector<Vec3> c2 = trackCtrl;
    c2.push_back(trackCtrl[0]);
    c2.push_back(trackCtrl[1]);
    SEGS = (int)c2.size() - 3;
    trackSamples.clear();
    for (int s = 0; s < SEGS; ++s) {
        for (int i = 0; i < SAMPLE_DENSITY / SEGS; ++i) {
            float t = (float)i / (SAMPLE_DENSITY / SEGS);
            trackSamples.push_back(catmull(c2[s], c2[s + 1], c2[s + 2], c2[s + 3], t));
        }
    }
    trackLength = 0.0f;
    for (size_t i = 1; i < trackSamples.size(); ++i) trackLength += length(trackSamples[i] - trackSamples[i - 1]);
    mapMinX = mapMaxX = trackSamples[0].x;
    mapMinZ = mapMaxZ = trackSamples[0].z;
    for (auto& p : trackSamples) {
        mapMinX = min(mapMinX, p.x);
        mapMaxX = max(mapMaxX, p.x);
        mapMinZ = min(mapMinZ, p.z);
        mapMaxZ = max(mapMaxZ, p.z);
    }
    float padX = (mapMaxX - mapMinX) * 0.05f;
    float padZ = (mapMaxZ - mapMinZ) * 0.05f;
    mapMinX -= padX;
    mapMaxX += padX;
    mapMinZ -= padZ;
    mapMaxZ += padZ;
    cerr << "[Track] len = " << trackLength << "u\n";
    // ---- 設定拱門 ----
    Vec3 dir0 = normalize(trackSamples[1] - trackSamples[0]);
    // 將起點拱門向前移動5個單位
    gateStartPos = trackSamples[0] + dir0 * 5.0f;
    gateStartHdg = atan2f(dir0.x, dir0.z) * 180.0f / M_PI;
    int midIdx = (int)trackSamples.size() - 1;  // 這是正確的
    gateFinishPos = trackSamples[midIdx];
    Vec3 dirF = normalize(trackSamples[(midIdx + 1) % trackSamples.size()] - gateFinishPos);
    gateFinishHdg = atan2f(dirF.x, dirF.z) * 180.0f / M_PI;
    // ---- 計算終點空間邊界 ----
    Vec3 finishVolumeCenter = gateFinishPos;
    Vec3 finishDirForward = dirF;  // 賽道前進方向
    Vec3 finishDirUp = {0.0f, 1.0f, 0.0f};
    Vec3 finishDirRight = normalize(cross(finishDirForward, finishDirUp));
    if (length(finishDirRight) < 1e-5) {                            // 如果dirF接近垂直，重新計算right
        finishDirRight = normalize(cross({0, 0, 1}, finishDirUp));  // 假設賽道主要在XZ平面
    }
    // 終點空間的8個角點 (先計算相對於中心的偏移)
    Vec3 halfSize = {FINISH_VOLUME_WIDTH / 2.0f, FINISH_VOLUME_HEIGHT / 2.0f, FINISH_VOLUME_DEPTH / 2.0f};
    float absDirFx = fabs(dirF.x);
    float absDirFz = fabs(dirF.z);

    float extentX = halfSize.x * absDirFz + halfSize.z * absDirFx;  // X方向延伸
    float extentZ = halfSize.x * absDirFx + halfSize.z * absDirFz;  // Z方向延伸

    finishVolumeMin.x = gateFinishPos.x - extentX;
    finishVolumeMax.x = gateFinishPos.x + extentX;
    finishVolumeMin.y = gateFinishPos.y;  // 從地面開始
    finishVolumeMax.y = gateFinishPos.y + FINISH_VOLUME_HEIGHT;
    finishVolumeMin.z = gateFinishPos.z - extentZ;
    finishVolumeMax.z = gateFinishPos.z + extentZ;
}
void drawFinishVolume() {
    // 繪製終點檢測空間 (用於調試)
    glPushMatrix();
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.0f, 1.0f, 0.0f, 0.2f);  // 半透明綠色

    // 使用 AABB 的邊界繪製
    // 終點綠色
    glBegin(GL_QUADS);
    // Bottom face
    glVertex3f(finishVolumeMin.x, finishVolumeMin.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMin.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMin.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMin.x, finishVolumeMin.y, finishVolumeMax.z);
    // Top face
    glVertex3f(finishVolumeMin.x, finishVolumeMax.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMax.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMax.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMin.x, finishVolumeMax.y, finishVolumeMax.z);
    // Front face
    glVertex3f(finishVolumeMin.x, finishVolumeMin.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMin.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMax.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMin.x, finishVolumeMax.y, finishVolumeMin.z);
    // Back face
    glVertex3f(finishVolumeMin.x, finishVolumeMin.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMin.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMax.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMin.x, finishVolumeMax.y, finishVolumeMax.z);
    // Left face
    glVertex3f(finishVolumeMin.x, finishVolumeMin.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMin.x, finishVolumeMin.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMin.x, finishVolumeMax.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMin.x, finishVolumeMax.y, finishVolumeMin.z);
    // Right face
    glVertex3f(finishVolumeMax.x, finishVolumeMin.y, finishVolumeMin.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMin.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMax.y, finishVolumeMax.z);
    glVertex3f(finishVolumeMax.x, finishVolumeMax.y, finishVolumeMin.z);
    glEnd();

    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);  // 如果場景中其他物件需要光照
    glPopMatrix();
}
void drawResultScreen() {
    /* 進 2D 正交投影 */
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    /* 繪製背景圖像 */
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texSettlement);
    glColor3f(1, 1, 1);  // 白色以顯示原始顏色
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2i(0, 0);
    glTexCoord2f(1, 0);
    glVertex2i(winW, 0);
    glTexCoord2f(1, 1);
    glVertex2i(winW, winH);
    glTexCoord2f(0, 1);
    glVertex2i(0, winH);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    /* 半透明覆蓋層 - 提高可讀性 */
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0, 0, 0, 0.5f);  // 半透明黑色
    glBegin(GL_QUADS);
    glVertex2i(0, 0);
    glVertex2i(winW, 0);
    glVertex2i(winW, winH);
    glVertex2i(0, winH);
    glEnd();

    /* 標題 */
    glColor3f(1, 1, 0);
    glRasterPos2i(winW / 2 - 80, winH / 2 + 100);
    const char* title = "RACE RESULTS";
    while (*title) glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *title++);

    /* 玩家結果表格 */
    // 表格位置和大小
    const int tableWidth = 500;
    const int rowHeight = 40;
    const int tableX = winW / 2 - tableWidth / 2;
    int tableY;
    if (previousGameMode == MODE_SOLO) {
        tableY = winH / 2;
    } else {
        tableY = winH / 2 - 40;
    }
    const char* headers[] = {"Rank", "Player", "Time", "Status"};
    const int colWidths[] = {80, 120, 150, 150};
    int colX = tableX;
    glColor3f(0.8f, 0.8f, 0.8f);
    for (int i = 0; i < 4; i++) {
        glRasterPos2i(colX + 10, tableY + 50);
        const char* header = headers[i];
        while (*header) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *header++);
        colX += colWidths[i];
    }
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);
    glVertex2i(tableX, tableY + 30);
    glVertex2i(tableX + tableWidth, tableY + 30);
    glEnd();

    // 計算出是誰第一、誰第二
    int firstPlayer = (finalRankP1 == 1) ? 1 : 2;
    int secondPlayer = (finalRankP1 == 1) ? 2 : 1;
    char buf[64];
    int playerCount = 0;

    // 玩家1和玩家2的數據
    struct PlayerData {
        int id;
        int rank;
        float time;
        bool finished;
        const char* color;  // "blue" 或 "red"
    } players[2];
    if (previousGameMode == MODE_SOLO) {
        // Solo mode only shows player 1
        players[0] = {1, 1, finalTimeP1, p1Finished, "blue"};  // Always rank 1 in solo
        playerCount = 1;
    } else {
        // PVP and AI mode show both players
        players[0] = {1, finalRankP1, finalTimeP1, p1Finished, "blue"};
        players[1] = {2, finalRankP2, finalTimeP2, p2Finished, "red"};
        if (players[0].rank > players[1].rank) {
            swap(players[0], players[1]);
        }
        playerCount = 2;
    }
    for (int i = 0; i < playerCount; i++) {
        colX = tableX;
        int rowY = tableY - i * rowHeight;
        if (strcmp(players[i].color, "blue") == 0) {
            glColor3f(0.2f, 0.2f, 1.0f);
        } else {
            glColor3f(1.0f, 0.2f, 0.2f);
        }
        // 排名
        sprintf(buf, "%d", players[i].rank);
        glRasterPos2i(colX + 10, rowY);
        char* text = buf;
        while (*text) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *text++);
        colX += colWidths[0];

        // 玩家名稱
        sprintf(buf, "Player %d", players[i].id);
        glRasterPos2i(colX + 10, rowY);
        text = buf;
        while (*text) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *text++);
        colX += colWidths[1];

        // 時間
        if (players[i].finished) {
            sprintf(buf, "%02d:%05.2f", (int)(players[i].time) / 60, fmod(players[i].time, 60));
        } else {
            sprintf(buf, "--:--");
        }
        glRasterPos2i(colX + 10, rowY);
        text = buf;
        while (*text) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *text++);
        colX += colWidths[2];

        // 狀態
        const char* status = players[i].finished ? "FINISHED" : "UNFINISHED";
        glRasterPos2i(colX + 10, rowY);
        text = (char*)status;
        while (*text) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *text++);
    }

    // 底部提示
    const char* tip = "Press SPACE to return to menu";
    glColor3f(1, 1, 1);
    glRasterPos2i(winW / 2 - 140, winH / 2 - 120);
    while (*tip) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *tip++);

    /* 還原 */
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ------------------------------------------------------------
// 繪製網格
// ------------------------------------------------------------
void drawGrid(float size = 7000, float step = 100) {
    glDisable(GL_LIGHTING);
    // ---- 繪製帶有紋理的草地 ----
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texGrass);
    glColor3f(1.0f, 1.0f, 1.0f);  // 設定為白色，才能完整顯示紋理的原色
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    const float textureRepeat = size / 50.0f;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-size, -0.1, -size);
    glTexCoord2f(textureRepeat, 0.0f);
    glVertex3f(size, -0.1, -size);
    glTexCoord2f(textureRepeat, textureRepeat);
    glVertex3f(size, -0.1, size);
    glTexCoord2f(0.0f, textureRepeat);
    glVertex3f(-size, -0.1, size);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    // ---- 繪製參考用的紅色和藍色軸線 ----
    // glLineWidth(3.0f);
    // glColor3f(1.0f, 0.0f, 0.0f);  // 紅色 X 軸
    // glBegin(GL_LINES);
    // glVertex3f(-size, 5.0f, 0.0f);
    // glVertex3f(size, 5.0f, 0.0f);
    // glEnd();
    // glColor3f(0.0f, 0.0f, 1.0f);  // 藍色 Z 軸
    // glBegin(GL_LINES);
    // glVertex3f(0.0f, 5.0f, -size);
    // glVertex3f(0.0f, 5.0f, size);
    // glEnd();

    glLineWidth(1.0f);  // 恢復預設線寬
}

// 繪製賽道
void drawTrack3D() {
    glDisable(GL_LIGHTING);
    // ---------- (a) 沙地 ----------
    glColor3f(1.0f, 1.0f, 1.0f);  // 設定為白色，避免對紋理造成色偏

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texSand);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glBegin(GL_QUAD_STRIP);
    float distanceAlongTrackSand = 0.0f;
    const float textureRepeatU_Sand = 6.0f;   // 沙地紋理在路寬上重複的次數 (可調整)
    const float textureRepeatV_Sand = 0.04f;  // 沙地紋理沿賽道長度方向的縮放比例 (可調整)

    for (size_t i = 0; i <= trackSamples.size(); ++i) {
        const Vec3& p = trackSamples[i % trackSamples.size()];
        const Vec3& q = trackSamples[(i + 1) % trackSamples.size()];

        if (i > 0) {
            const Vec3& p_prev = trackSamples[(i - 1 + trackSamples.size()) % trackSamples.size()];
            distanceAlongTrackSand += length(p - p_prev);
        }
        Vec3 dir = normalize(q - p);
        Vec3 n = {-dir.z, 0, dir.x};
        glNormal3f(0.0f, 1.0f, 0.0f);  // 設定沙地表面的法線 (朝上)
        glTexCoord2f(textureRepeatU_Sand, distanceAlongTrackSand * textureRepeatV_Sand);
        glVertex3f((p + n * SAND_HALF_W).x, p.y + 0.003f, (p + n * SAND_HALF_W).z);
        glTexCoord2f(0.0f, distanceAlongTrackSand * textureRepeatV_Sand);
        glVertex3f((p - n * SAND_HALF_W).x, p.y + 0.003f, (p - n * SAND_HALF_W).z);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);  // 關閉沙地的紋理貼圖
    // ---------- (b) 柏油路 ----------
    glDisable(GL_LIGHTING);       // 確保光照啟用，讓紋理有明暗變化
    glColor3f(1.0f, 1.0f, 1.0f);  // 將顏色設為白色，才不會對紋理造成色偏

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texAsphalt);
    // 讓紋理顏色與光照顏色混合
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBegin(GL_QUAD_STRIP);
    float distanceAlongTrack = 0.0f;
    // 調整這兩個值可以改變紋理重複的密度
    const float textureRepeatU = 4.0f;   // 紋理在路寬上重複的次數
    const float textureRepeatV = 0.05f;  // 紋理沿賽道長度方向的縮放比例

    for (size_t i = 0; i <= trackSamples.size(); ++i) {
        const Vec3& p = trackSamples[i % trackSamples.size()];
        const Vec3& q = trackSamples[(i + 1) % trackSamples.size()];

        // 計算 V 座標 (沿著賽道方向)
        if (i > 0) {
            const Vec3& p_prev = trackSamples[(i - 1 + trackSamples.size()) % trackSamples.size()];
            distanceAlongTrack += length(p - p_prev);
        }

        Vec3 dir = normalize(q - p);
        Vec3 n = {-dir.z, 0, dir.x};
        glNormal3f(0.0f, 1.0f, 0.0f);
        // 右側頂點 (U = textureRepeatU)
        glTexCoord2f(textureRepeatU, distanceAlongTrack * textureRepeatV);
        glVertex3f((p + n * ROAD_HALF_W).x, p.y + 0.01f, (p + n * ROAD_HALF_W).z);

        // 左側頂點 (U = 0)
        glTexCoord2f(0.0f, distanceAlongTrack * textureRepeatV);
        glVertex3f((p - n * ROAD_HALF_W).x, p.y + 0.01f, (p - n * ROAD_HALF_W).z);
    }
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

// ---------- 紅白相間的路緣石 ----------
void drawCurbstones() {
    const float CURB_WIDTH = 0.8f;                   // 路緣石寬度
    const float CURB_HEIGHT = 0.02f;                 // 增加路緣石高度，更加突出
    const float SEGMENT_LENGTH = 2.0f;               // 每段紅白相間的長度
    const float BEVEL_SIZE = 0.05f;                  // 添加斜角尺寸
    glPushAttrib(GL_LIGHTING_BIT | GL_POLYGON_BIT);  // 保存光照和多邊形相關屬性（包含著色模式）

    glDisable(GL_LIGHTING);
    glShadeModel(GL_FLAT);  // 設定為平面著色，實現紅白分明

    // 遍歷賽道點，繪製路緣石
    float accumulatedLength = 0.0f;  // 累積的路徑長度，用於切換顏色
    bool isRed = true;               // 起始顏色為紅

    // 繪製兩側路緣石
    for (int side = -1; side <= 1; side += 2) {  // -1是左側, +1是右側
        accumulatedLength = 0.0f;
        isRed = true;

        // 使用GL_QUAD_STRIP繪製路緣石頂面
        glBegin(GL_QUAD_STRIP);

        for (size_t i = 0; i <= trackSamples.size(); ++i) {
            const Vec3& p = trackSamples[i % trackSamples.size()];
            const Vec3& q = trackSamples[(i + 1) % trackSamples.size()];
            Vec3 dir = normalize(q - p);
            Vec3 n = {-dir.z, 0, dir.x};
            Vec3 innerEdge = p + n * side * ROAD_HALF_W;
            Vec3 outerEdge = p + n * side * (ROAD_HALF_W + CURB_WIDTH);

            // 根據累積長度切換顏色
            if (i > 0) {
                const Vec3& prevP = trackSamples[(i - 1) % trackSamples.size()];
                accumulatedLength += length(p - prevP);
                if (accumulatedLength >= SEGMENT_LENGTH) {
                    isRed = !isRed;
                    accumulatedLength -= SEGMENT_LENGTH;
                }
            }

            // 設置顏色 - 添加光澤效果
            if (isRed) {
                glColor3f(0.95f, 0.25f, 0.25f);  // 亮紅色
            } else {
                glColor3f(1.0f, 1.0f, 1.0f);  // 白色
            }
            glVertex3f(outerEdge.x, outerEdge.y + CURB_HEIGHT, outerEdge.z);  // 外側頂部
            glVertex3f(innerEdge.x, innerEdge.y + CURB_HEIGHT, innerEdge.z);  // 內側頂部
        }
        glEnd();

        // 繪製路緣石外側面
        accumulatedLength = 0.0f;
        isRed = true;

        glBegin(GL_QUAD_STRIP);
        for (size_t i = 0; i <= trackSamples.size(); ++i) {
            const Vec3& p = trackSamples[i % trackSamples.size()];
            const Vec3& q = trackSamples[(i + 1) % trackSamples.size()];

            // 計算當前點與下一點之間的方向向量和法向量
            Vec3 dir = normalize(q - p);
            Vec3 n = {-dir.z, 0, dir.x};

            // 計算路緣石外側面頂點
            Vec3 outerEdgeTop = p + n * side * (ROAD_HALF_W + CURB_WIDTH);
            outerEdgeTop.y += CURB_HEIGHT;

            Vec3 outerEdgeBottom = p + n * side * (ROAD_HALF_W + CURB_WIDTH);
            outerEdgeBottom.y += 0.01f;  // 略高於地面

            // 根據累積長度切換顏色
            if (i > 0) {
                const Vec3& prevP = trackSamples[(i - 1) % trackSamples.size()];
                accumulatedLength += length(p - prevP);
                if (accumulatedLength >= SEGMENT_LENGTH) {
                    isRed = !isRed;
                    accumulatedLength -= SEGMENT_LENGTH;
                }
            }

            // 設置側面顏色 - 使用漸變效果增強立體感
            if (isRed) {
                glColor3f(0.95f, 0.25f, 0.25f);  // 設定為與頂面相同的亮紅色
            } else {
                glColor3f(1.0f, 1.0f, 1.0f);  // 設定為與頂面相同的白色
            }
            glVertex3f(outerEdgeTop.x, outerEdgeTop.y, outerEdgeTop.z);
            glVertex3f(outerEdgeBottom.x, outerEdgeBottom.y, outerEdgeBottom.z);
        }
        glEnd();

        // 繪製路緣石內側面 (斜面)
        accumulatedLength = 0.0f;
        isRed = true;

        glBegin(GL_QUAD_STRIP);
        for (size_t i = 0; i <= trackSamples.size(); ++i) {
            const Vec3& p = trackSamples[i % trackSamples.size()];
            const Vec3& q = trackSamples[(i + 1) % trackSamples.size()];

            // 計算當前點與下一點之間的方向向量和法向量
            Vec3 dir = normalize(q - p);
            Vec3 n = {-dir.z, 0, dir.x};

            // 計算路緣石內側面頂點 (斜面)
            Vec3 innerEdgeTop = p + n * side * ROAD_HALF_W;
            innerEdgeTop.y += CURB_HEIGHT;

            Vec3 innerEdgeBottom = p + n * side * (ROAD_HALF_W + BEVEL_SIZE);
            innerEdgeBottom.y += 0.01f;  // 略高於路面

            // 根據累積長度切換顏色
            if (i > 0) {
                const Vec3& prevP = trackSamples[(i - 1) % trackSamples.size()];
                accumulatedLength += length(p - prevP);
                if (accumulatedLength >= SEGMENT_LENGTH) {
                    isRed = !isRed;
                    accumulatedLength -= SEGMENT_LENGTH;
                }
            }

            // 設置內側斜面顏色
            if (isRed) {
                glColor3f(0.95f, 0.25f, 0.25f);  // 設定為與頂面相同的亮紅色
            } else {
                glColor3f(1.0f, 1.0f, 1.0f);  // 設定為與頂面相同的白色
            }
            glVertex3f(innerEdgeTop.x, innerEdgeTop.y, innerEdgeTop.z);
            glVertex3f(innerEdgeBottom.x, innerEdgeBottom.y, innerEdgeBottom.z);
        }
        glEnd();
    }

    // 添加路緣石轉角處的額外細節 (重點路段)
    for (size_t i = 0; i < trackSamples.size(); i += 30) {  // 每隔30個點檢查一次
        const Vec3& p0 = trackSamples[i];
        const Vec3& p1 = trackSamples[(i + 5) % trackSamples.size()];
        const Vec3& p2 = trackSamples[(i + 10) % trackSamples.size()];

        // 計算曲率 - 檢查是否為彎角
        Vec3 dir1 = normalize(p1 - p0);
        Vec3 dir2 = normalize(p2 - p1);
        float dot = dir1.x * dir2.x + dir1.z * dir2.z;  // 點積，越小彎度越大

        if (dot < 0.9f) {  // 如果彎度足夠大
            // 繪製反光標記 (在路緣石上方)
            for (int side = -1; side <= 1; side += 2) {
                const Vec3& p = p1;  // 使用中間點
                const Vec3& q = trackSamples[(i + 6) % trackSamples.size()];
                Vec3 dir = normalize(q - p);
                Vec3 n = {-dir.z, 0, dir.x};

                // 計算反光標記位置
                Vec3 markerPos = p + n * side * (ROAD_HALF_W + CURB_WIDTH / 2.0f);
                markerPos.y += CURB_HEIGHT + 0.05f;

                // 繪製反光標記 (紅色小三角形)
                glBegin(GL_TRIANGLES);
                glColor3f(1.0f, 0.3f, 0.1f);  // 鮮亮的橘紅色

                // 三角形指向彎內側
                float markerSize = 0.5f;
                Vec3 markerTip = markerPos + n * side * markerSize;
                Vec3 markerBase1 = markerPos + dir * (markerSize / 2.0f);
                Vec3 markerBase2 = markerPos - dir * (markerSize / 2.0f);

                glVertex3f(markerTip.x, markerTip.y, markerTip.z);
                glVertex3f(markerBase1.x, markerBase1.y, markerBase1.z);
                glVertex3f(markerBase2.x, markerBase2.y, markerBase2.z);
                glEnd();
            }
        }
    }
    glPopAttrib();
}

void initializeTrafficCones() {
    trafficCones.clear();
    if (trackSamples.empty() || trackSamples.size() < 2) return;
    srand(static_cast<unsigned int>(time(0)));
    const float RANDOM_CONE_SCALE_BASE = 0.025f;  // 三角錐基礎大小 (原 0.04f)
    int regularConesToPlace = 40;
    for (int i = 0; i < regularConesToPlace; ++i) {
        TrafficCone cone;
        int sampleIndex = (rand() % (trackSamples.size() - 40)) + 20;  // 避免太靠近起點或終點的固定排列區
        if (sampleIndex >= trackSamples.size() || sampleIndex < 0) sampleIndex = trackSamples.size() / 2;

        const Vec3& p1 = trackSamples[sampleIndex];
        const Vec3& p2 = trackSamples[(sampleIndex + 1) % trackSamples.size()];
        Vec3 trackDir = normalize(p2 - p1);
        Vec3 normalDir = {-trackDir.z, 0.0f, trackDir.x};
        float side = (rand() % 2 == 0) ? 1.0f : -1.0f;
        // 確保隨機放置的三角錐在沙地區域或更外側一點
        float offset_min_rand = ROAD_HALF_W + 1.0f;  // 至少在路肩外
        float offset_max_rand = SAND_HALF_W + 5.0f;  // 可以稍微超出沙地一點
        float offset = offset_min_rand + static_cast<float>(rand()) /
                                             (static_cast<float>(RAND_MAX / (offset_max_rand - offset_min_rand)));

        cone.pos = p1 + normalDir * side * offset;
        // 根據橫向距離決定Y值
        float lateralDistFromCenter = length(cone.pos - p1);  // 近似橫向距離
        if (lateralDistFromCenter <= ROAD_HALF_W) {
            cone.pos.y = p1.y + 0.01f;  // 柏油路高度
        } else if (lateralDistFromCenter <= SAND_HALF_W) {
            cone.pos.y = p1.y + 0.003f;  // 沙地高度
        } else {
            cone.pos.y = p1.y - 0.05f;  // 草地/更外側，略低於賽道基準 (p1.y通常為0)
        }

        cone.isHit = false;
        cone.isActive = true;
        cone.velocity = {0, 0, 0};
        cone.angularVelocity = {0, 0, 0};
        cone.rotationAngle = static_cast<float>(rand() % 360);
        cone.rotationAxis = {0.0f, 1.0f, 0.0f};
        cone.scale = RANDOM_CONE_SCALE_BASE;
        cone.model = &Cone1;

        bool canPlace = true;
        for (const auto& placedCone : trafficCones) {
            if (length(cone.pos - placedCone.pos) <
                CONE_COLLISION_RADIUS_CONE * 1.5f * (cone.scale + placedCone.scale)) {  // 稍微加大排斥距離
                canPlace = false;
                break;
            }
        }
        if (canPlace) {
            trafficCones.push_back(cone);
        }
    }

    // --- 在起點線後方隨機放置一排延伸至沙地的三角錐 ---
    const int NUM_START_LINE_CONES = 20;
    // 讓三角錐分佈的總寬度略小於沙地總寬度，確保它們在沙地內或邊緣
    const float START_LINE_TOTAL_WIDTH = (SAND_HALF_W - 1.0f) * 2.0f;  // 例如，沙地半寬減去一點邊界再乘以2
    const float START_LINE_CONE_SPACING =
        START_LINE_TOTAL_WIDTH / (NUM_START_LINE_CONES > 1 ? (NUM_START_LINE_CONES - 1) : 1);
    const float START_LINE_CONE_OFFSET_BEHIND_START = -25.0f;                   // 放置在起點線後方遠一點
    const float START_LINE_POS_RANDOM_FACTOR = START_LINE_CONE_SPACING * 0.4f;  // 位置隨機擾動幅度
    const float START_LINE_ANGLE_RANDOM_FACTOR = 25.0f;                         // 角度隨機擾動幅度 (度)
    const float START_LINE_SCALE_BASE = 0.03f;
    const float START_LINE_SCALE_RANGE = 0.01f;

    if (trackSamples.size() >= 2) {
        Vec3 startPointActual = trackSamples[0];  // 賽道實際起點 (y=0)
        Vec3 dirToNextPoint = normalize(trackSamples[1] - startPointActual);
        Vec3 perpendicularDir = {-dirToNextPoint.z, 0.0f, dirToNextPoint.x};

        Vec3 rowCenterPos = startPointActual + dirToNextPoint * START_LINE_CONE_OFFSET_BEHIND_START;
        for (int i = 0; i < NUM_START_LINE_CONES; ++i) {
            TrafficCone cone;
            float baseOffsetFromCenter =
                (static_cast<float>(i) - static_cast<float>(NUM_START_LINE_CONES - 1) / 2.0f) * START_LINE_CONE_SPACING;

            // 添加位置隨機性
            float randomX_offset = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * START_LINE_POS_RANDOM_FACTOR;
            float randomZ_offset = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * START_LINE_POS_RANDOM_FACTOR;

            cone.pos = rowCenterPos + perpendicularDir * baseOffsetFromCenter;
            cone.pos.x += randomX_offset;
            cone.pos.z += randomZ_offset;
            // 根據最終的X,Z位置判斷其Y高度
            float lateralDistFromStartCenterline = abs((cone.pos.x - startPointActual.x) * perpendicularDir.x +
                                                       (cone.pos.z - startPointActual.z) * perpendicularDir.z);
            float distToCenter = lateralOffsetFromCenter({cone.pos.x, startPointActual.y, cone.pos.z});
            if (distToCenter <= ROAD_HALF_W) {
                cone.pos.y = startPointActual.y + 0.01f;   // 柏油路高度
            } else {                                       // ROAD_HALF_W < distToCenter <= SAND_HALF_W (或更外)
                cone.pos.y = startPointActual.y + 0.003f;  // 沙地高度
            }
            cone.isHit = false;
            cone.isActive = true;
            cone.velocity = {0, 0, 0};
            cone.angularVelocity = {0, 0, 0};
            // 添加角度隨機性
            cone.rotationAngle = (atan2f(dirToNextPoint.x, dirToNextPoint.z) * 180.0f / M_PI) +
                                 (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * START_LINE_ANGLE_RANDOM_FACTOR;
            cone.rotationAxis = {0.0f, 1.0f, 0.0f};
            cone.scale = START_LINE_SCALE_BASE +
                         static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / START_LINE_SCALE_RANGE));
            cone.model = &Cone1;

            trafficCones.push_back(cone);
        }
    }

    cout << "[INFO] Total initialized " << trafficCones.size() << " traffic cones." << std::endl;
}

void drawTrafficCones() {
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    const float CONE_MODEL_HALF_HEIGHT_UNSCALED = 18.0f;
    for (const auto& cone : trafficCones) {
        if (!cone.isActive || !cone.model) continue;
        glPushMatrix();
        float vertical_offset = CONE_MODEL_HALF_HEIGHT_UNSCALED * cone.scale;
        glTranslatef(cone.pos.x, cone.pos.y + vertical_offset, cone.pos.z);  // Y軸平移修正
        glRotatef(cone.rotationAngle, cone.rotationAxis.x, cone.rotationAxis.y, cone.rotationAxis.z);
        glScalef(cone.scale, cone.scale, cone.scale);
        drawMultiMaterialModel(*(cone.model));
        glPopMatrix();
    }
}

Vec3 slerpQuaternionAxis(const Vec3& axis_from_norm, float angle_from_deg, const Vec3& axis_to_norm, float angle_to_deg,
                         float t, float& out_angle_deg) {
    // 將軸角轉換為四元數
    float angle_from_rad = angle_from_deg * M_PI / 180.0f;
    float angle_to_rad = angle_to_deg * M_PI / 180.0f;

    Vec3 q_from_xyz = axis_from_norm * sinf(angle_from_rad / 2.0f);
    float q_from_w = cosf(angle_from_rad / 2.0f);

    Vec3 q_to_xyz = axis_to_norm * sinf(angle_to_rad / 2.0f);
    float q_to_w = cosf(angle_to_rad / 2.0f);
    float dot_product =
        q_from_xyz.x * q_to_xyz.x + q_from_xyz.y * q_to_xyz.y + q_from_xyz.z * q_to_xyz.z + q_from_w * q_to_w;
    if (dot_product < 0.0f) {
        q_to_xyz.x = -q_to_xyz.x;
        q_to_xyz.y = -q_to_xyz.y;
        q_to_xyz.z = -q_to_xyz.z;
        q_to_w = -q_to_w;
        dot_product = -dot_product;
    }
    float k0, k1;
    if (dot_product > 0.9995f) {  // 角度非常小，線性插值以避免除零
        k0 = 1.0f - t;
        k1 = t;
    } else {
        float sin_omega = sqrtf(1.0f - dot_product * dot_product);
        float omega = atan2f(sin_omega, dot_product);
        k0 = sinf((1.0f - t) * omega) / sin_omega;
        k1 = sinf(t * omega) / sin_omega;
    }
    Vec3 q_res_xyz;
    q_res_xyz.x = q_from_xyz.x * k0 + q_to_xyz.x * k1;
    q_res_xyz.y = q_from_xyz.y * k0 + q_to_xyz.y * k1;
    q_res_xyz.z = q_from_xyz.z * k0 + q_to_xyz.z * k1;
    float q_res_w = q_from_w * k0 + q_to_w * k1;

    // 將結果四元數轉換回軸角
    float sin_half_angle_sq = q_res_xyz.x * q_res_xyz.x + q_res_xyz.y * q_res_xyz.y + q_res_xyz.z * q_res_xyz.z;
    if (sin_half_angle_sq <= 0.00001f) {  // 幾乎是單位四元數（無旋轉）或非常小的旋轉
        out_angle_deg = 0.0f;
        return {0.0f, 1.0f, 0.0f};  // 預設軸
    }
    float sin_half_angle = sqrtf(sin_half_angle_sq);
    out_angle_deg = 2.0f * atan2f(sin_half_angle, q_res_w) * 180.0f / M_PI;
    if (out_angle_deg < 0) out_angle_deg += 360.0f;

    Vec3 out_axis = q_res_xyz * (1.0f / sin_half_angle);
    return normalize(out_axis);
}

void updateTrafficCones(float dt) {
    const float SETTLE_SPEED_THRESHOLD_SQR = CONE_MIN_VELOCITY_SQR * 25.0f;  // 更早開始安定過程的閾值 (速度平方)
    const float SETTLE_ANGULAR_THRESHOLD_SQR = CONE_MIN_ANGULAR_VELOCITY_SQR * 25.0f;  // (角速度平方)
    const float SETTLE_RATE_POS = 0.2f;                                                // 位置安定速率 (每幀插值比例)
    const float SETTLE_RATE_ROT = 0.15f;                                               // 旋轉安定速率

    for (auto& cone : trafficCones) {
        if (!cone.isActive) continue;
        if (cone.isHit) {
            // 1. 基本物理更新 (位置、速度、重力、地面碰撞)
            cone.pos = cone.pos + cone.velocity * dt;
            cone.velocity.y += CONE_GRAVITY * dt;
            if (cone.pos.y < 0.0f) {
                cone.pos.y = 0.0f;
                cone.velocity.y *= -0.4f;
                cone.velocity.x *= 0.8f;  // 地面摩擦
                cone.velocity.z *= 0.8f;
                cone.angularVelocity = cone.angularVelocity * 0.7f;  // 地面碰撞導致角速度衰減
            }
            // 2. 更新旋轉 (基於角速度)
            float angleChange_deg = length(cone.angularVelocity) * dt * (180.0f / M_PI);
            if (length(cone.angularVelocity) > 0.001f) {
                Vec3 current_rotation_axis_norm = normalize(cone.rotationAxis);
                if (length(current_rotation_axis_norm) < 0.001f) current_rotation_axis_norm = {0.0f, 1.0f, 0.0f};

                Vec3 angVel_norm = normalize(cone.angularVelocity);
                if (length(angVel_norm) < 0.001f) angVel_norm = current_rotation_axis_norm;
                cone.rotationAxis = angVel_norm;
                cone.rotationAngle += angleChange_deg;
                cone.rotationAngle = fmod(cone.rotationAngle, 360.0f);
            }
            // 3. 速度和角速度衰減 (空氣阻力/摩擦)
            cone.velocity = cone.velocity * pow(CONE_FRICTION, dt * 60.0f);
            cone.angularVelocity = cone.angularVelocity * pow(CONE_ANGULAR_FRICTION, dt * 60.0f);
            // 4. 判斷是否進入/處於安定階段或完全停止
            bool isAlmostStopped =
                length(cone.velocity) * length(cone.velocity) < CONE_MIN_VELOCITY_SQR &&
                length(cone.angularVelocity) * length(cone.angularVelocity) < CONE_MIN_ANGULAR_VELOCITY_SQR;
            bool isSettling =
                length(cone.velocity) * length(cone.velocity) < SETTLE_SPEED_THRESHOLD_SQR &&
                length(cone.angularVelocity) * length(cone.angularVelocity) < SETTLE_ANGULAR_THRESHOLD_SQR;
            if (isAlmostStopped && cone.pos.y <= 0.01f) {
                // 完全停止，強制設定最終姿態
                cone.isHit = false;
                cone.velocity = {0, 0, 0};
                cone.angularVelocity = {0, 0, 0};
                Vec3 local_y_axis = {0.0f, 1.0f, 0.0f};
                Vec3 current_axis_normalized = normalize(cone.rotationAxis);
                if (length(current_axis_normalized) < 0.001f) current_axis_normalized = {0.0f, 1.0f, 0.0f};
                float angle_rad = cone.rotationAngle * M_PI / 180.0f;
                float cos_a = cosf(angle_rad);
                float sin_a = sinf(angle_rad);
                Vec3 k = current_axis_normalized;
                Vec3 v = local_y_axis;
                Vec3 rotated_y_axis;
                rotated_y_axis.x = v.x * cos_a + (k.y * v.z - k.z * v.y) * sin_a +
                                   k.x * (k.x * v.x + k.y * v.y + k.z * v.z) * (1 - cos_a);
                rotated_y_axis.y = v.y * cos_a + (k.z * v.x - k.x * v.z) * sin_a +
                                   k.y * (k.x * v.x + k.y * v.y + k.z * v.z) * (1 - cos_a);
                rotated_y_axis.z = v.z * cos_a + (k.x * v.y - k.y * v.x) * sin_a +
                                   k.z * (k.x * v.x + k.y * v.y + k.z * v.z) * (1 - cos_a);
                float vertical_alignment = rotated_y_axis.y;
                const float UPRIGHT_COS_THRESHOLD = cosf(50.0f * M_PI / 180.0f);  // 閾值：50度以内算直立

                if (vertical_alignment > UPRIGHT_COS_THRESHOLD) {
                    cone.pos.y = 0.0f;
                    cone.rotationAxis = {0.0f, 1.0f, 0.0f};
                    cone.rotationAngle = static_cast<float>(rand() % 360);
                } else {
                    Vec3 random_horizontal_axis =
                        normalize({(float)rand() / RAND_MAX - 0.5f, 0.0f, (float)rand() / RAND_MAX - 0.5f});
                    if (length(random_horizontal_axis) < 0.001f) random_horizontal_axis = {1.0f, 0.0f, 0.0f};
                    cone.rotationAxis = random_horizontal_axis;
                    cone.rotationAngle = 120.0f;
                    cone.pos.y = CONE_COLLISION_RADIUS_CONE * cone.scale * 0.15f;  // 視覺調整值
                    if (cone.pos.y < 0.001f) cone.pos.y = 0.001f;
                }

            } else if (isSettling && cone.pos.y <= 0.1f) {
                // 接近停止，開始平滑過渡到目標姿態
                Vec3 local_y_axis = {0.0f, 1.0f, 0.0f};
                Vec3 current_axis_normalized = normalize(cone.rotationAxis);
                if (length(current_axis_normalized) < 0.001f) current_axis_normalized = {0.0f, 1.0f, 0.0f};
                float angle_rad = cone.rotationAngle * M_PI / 180.0f;
                float cos_a = cosf(angle_rad);
                float sin_a = sinf(angle_rad);
                Vec3 k_rot = current_axis_normalized;  // 使用 k_rot 避免與 Vec3 k 重複宣告
                Vec3 v_rot = local_y_axis;             // 使用 v_rot
                Vec3 rotated_y_axis;
                rotated_y_axis.x = v_rot.x * cos_a + (k_rot.y * v_rot.z - k_rot.z * v_rot.y) * sin_a +
                                   k_rot.x * (k_rot.x * v_rot.x + k_rot.y * v_rot.y + k_rot.z * v_rot.z) * (1 - cos_a);
                rotated_y_axis.y = v_rot.y * cos_a + (k_rot.z * v_rot.x - k_rot.x * v_rot.z) * sin_a +
                                   k_rot.y * (k_rot.x * v_rot.x + k_rot.y * v_rot.y + k_rot.z * v_rot.z) * (1 - cos_a);
                rotated_y_axis.z = v_rot.z * cos_a + (k_rot.x * v_rot.y - k_rot.y * v_rot.x) * sin_a +
                                   k_rot.z * (k_rot.x * v_rot.x + k_rot.y * v_rot.y + k_rot.z * v_rot.z) * (1 - cos_a);
                float vertical_alignment = rotated_y_axis.y;
                const float UPRIGHT_COS_THRESHOLD = cosf(50.0f * M_PI / 180.0f);  // 閾值：50度以内算直立

                Vec3 targetRotationAxis;
                float targetRotationAngle_deg;
                float targetPosY;

                if (vertical_alignment > UPRIGHT_COS_THRESHOLD) {  // 目標：直立
                    targetRotationAxis = {0.0f, 1.0f, 0.0f};
                    targetRotationAngle_deg = cone.rotationAngle;   // 理想情況下，這個角度應該是只繞Y軸的部分
                    if (fabs(current_axis_normalized.y) < 0.95f) {  // 如果不是主要繞Y軸
                        targetRotationAngle_deg = static_cast<float>(rand() % 360);  // 給一個隨機Y軸目標角
                    }
                    targetPosY = 0.0f;
                } else {  // 目標：傾倒
                    targetRotationAxis =
                        normalize({(float)rand() / RAND_MAX - 0.5f, 0.0f, (float)rand() / RAND_MAX - 0.5f});
                    if (length(targetRotationAxis) < 0.001f) targetRotationAxis = {1.0f, 0.0f, 0.0f};
                    targetRotationAngle_deg = 120.0f;
                    targetPosY = CONE_COLLISION_RADIUS_CONE * cone.scale * 0.15f;
                    if (targetPosY < 0.001f) targetPosY = 0.001f;
                }

                // 平滑插值到目標姿態
                float settleFactorRot = SETTLE_RATE_ROT * dt * 60.0f;  // dt補償
                settleFactorRot = min(settleFactorRot, 1.0f);          // 避免過度插值

                float interpolatedAngle_deg;
                Vec3 interpolatedAxis =
                    slerpQuaternionAxis(current_axis_normalized, cone.rotationAngle, targetRotationAxis,
                                        targetRotationAngle_deg, settleFactorRot, interpolatedAngle_deg);

                cone.rotationAxis = interpolatedAxis;
                cone.rotationAngle = interpolatedAngle_deg;

                float settleFactorPos = SETTLE_RATE_POS * dt * 60.0f;
                settleFactorPos = min(settleFactorPos, 1.0f);
                cone.pos.y += (targetPosY - cone.pos.y) * settleFactorPos;

                // 速度和角速度也更快地衰減到0
                cone.velocity = cone.velocity * 0.85f;  // 加速衰減
                cone.angularVelocity = cone.angularVelocity * 0.85f;
            }
        }
    }
}
void handleCarConeCollision(Car& carToCollide, float dt) {
    if (trafficCones.empty()) return;
    Vec3 carPos = {carToCollide.x, carToCollide.y, carToCollide.z};
    for (auto& cone : trafficCones) {
        if (!cone.isActive || cone.isHit) continue;  // 如果不活動或已經被撞飛，則跳過

        Vec3 coneCenterPos = cone.pos;
        coneCenterPos.y += CONE_COLLISION_RADIUS_CONE * cone.scale;  // 假設三角錐的碰撞中心在其底部往上一點

        float distSq = (carPos.x - coneCenterPos.x) * (carPos.x - coneCenterPos.x) +
                       (carPos.z - coneCenterPos.z) * (carPos.z - coneCenterPos.z);

        float combinedRadius = CAR_COLLISION_RADIUS + CONE_COLLISION_RADIUS_CONE * cone.scale;

        if (distSq < combinedRadius * combinedRadius) {
            cone.isHit = true;
            Vec3 collisionDir = normalize(coneCenterPos - carPos);
            if (length(collisionDir) < 0.001f) {  // 如果中心重疊，隨機一個方向
                collisionDir = {static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f, 0.0f,
                                static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f};
                collisionDir = normalize(collisionDir);
                if (length(collisionDir) < 0.001f) collisionDir = {1, 0, 0};  // 最後的備用
            }
            float impulseStrength =
                abs(carToCollide.speed) * 0.3f +
                (CONE_HIT_IMPULSE_MIN +
                 static_cast<float>(rand()) / (RAND_MAX / (CONE_HIT_IMPULSE_MAX - CONE_HIT_IMPULSE_MIN)));

            cone.velocity = collisionDir * impulseStrength;
            cone.velocity.y += impulseStrength * 0.3f;  // 給一個向上的初速度，使其飛起來

            // 給予隨機的角速度
            cone.angularVelocity.x = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 360.0f * 2.0f;  // 度/秒
            cone.angularVelocity.y = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 360.0f * 2.0f;
            cone.angularVelocity.z = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 360.0f * 2.0f;

            // 為了更真實，角速度的方向可以基於碰撞點和碰撞方向
            // 這裡用簡化的隨機值
            cone.rotationAxis = normalize(
                {(float)rand() / RAND_MAX - 0.5f, (float)rand() / RAND_MAX - 0.5f, (float)rand() / RAND_MAX - 0.5f});
            if (length(cone.rotationAxis) < 0.001f) cone.rotationAxis = {0, 1, 0};

            // 可選：對車輛施加一個小的反作用力或減速
            // carToCollide.speed *= 0.98f;
        }
    }
}

// ---------- 賽道圍欄：在沙地外圍 ----------
// 繪製護欄組件
void drawTrackBarriers() {
    // 更新後的常數
    const float BARRIER_POST_HEIGHT = 1.8f;         // 立柱高度
    const float POST_XZ_THICKNESS = 0.20f;          // 立柱的XZ厚度 (使其為方形柱)
    const float POST_INTERVAL = 7.0f;               // 立柱間隔
    const float LOWER_RAIL_CENTER_Y_OFFSET = 0.6f;  // 下層護欄中心點離地高度
    const float UPPER_RAIL_CENTER_Y_OFFSET = 1.2f;  // 上層護欄中心點離地高度
    const float RAIL_BEAM_HEIGHT = 0.25f;           // 護欄橫樑本身的垂直高度
    const float RAIL_BEAM_DEPTH = 0.08f;            // 護欄橫樑本身的深度 (厚度)
    const float BARRIER_OFFSET_FROM_SAND = 0.5f;    // 圍欄距離沙地邊緣的偏移量
    const float REFLECTOR_SIZE = 0.15f;             // 反光片大小
    const float REFLECTOR_HEIGHT_OFFSET = 0.8f;     // 反光片中心離地高度

    glDisable(GL_LIGHTING);  // 確保在手動設定顏色時不受光照影響

    // 遍歷賽道點，每隔一定距離放置一個圍欄單元
    Vec3 lastTrackSamplePos = trackSamples[0];  // 用於計算賽段方向

    // 繪製兩側圍欄
    for (int side = -1; side <= 1; side += 2) {          // -1是左側, +1是右側
        float accumulatedDistanceOnSide = 0.0f;          // 用於紅白條紋的累計距離
        Vec3 prevPostBasePosForColor = trackSamples[0];  // 用於計算紅白條紋的起始點
        int postCounterForReflector = 0;                 // 用於間隔繪製反光片

        for (size_t i = 0; i < trackSamples.size(); ++i) {  // 遍歷所有賽道取樣點作為潛在的立柱或護欄段起點
            const Vec3& currentTrackSamplePos = trackSamples[i];
            const Vec3& nextTrackSamplePos = trackSamples[(i + 1) % trackSamples.size()];

            Vec3 segmentDir = normalize(nextTrackSamplePos - currentTrackSamplePos);
            Vec3 segmentNormal = {-segmentDir.z, 0, segmentDir.x};  // 路段的法向量

            float segmentLength = length(nextTrackSamplePos - currentTrackSamplePos);
            if (segmentLength < 0.1f) continue;  // 太短的賽段跳過

            // 在當前賽段上放置立柱
            // 從 accumulatedDistanceOnSide 與 POST_INTERVAL 的關係決定第一個立柱是否在本賽段上
            // 或者簡化：每個 track sample 點附近都考慮放柱子，但要確保間隔
            // 這裡的邏輯需要更精確地沿著賽道總長度來放置立柱，
            // 目前的寫法是基於 trackSamples 的索引，可能導致立柱分佈不均勻。
            // 為了簡化修改，我們先保留原有的基於 POST_INTERVAL 的邏輯，但顏色部分改進。

            // --- 簡化立柱與護欄生成邏輯，專注於外觀 ---
            // 假設每隔 POST_INTERVAL 放置一個立柱
            // 我們需要一個更連續的方式來放置立柱，而不僅僅是基於當前處理的 trackSamples[i]
            // 為了演示，我們將在每個 trackSamples[i] 處（如果它與上一個立柱足夠遠）放置立柱
            // 這不是最優的放置方式，但足以展示外觀。

            // 計算當前點作為立柱基座的位置
            Vec3 postBasePos = currentTrackSamplePos;
            Vec3 postCenterPos = postBasePos + segmentNormal * side * (SAND_HALF_W + BARRIER_OFFSET_FROM_SAND);
            postCenterPos.y = postBasePos.y;  // 確保Y軸在地面

            // 計算到下一個理論立柱位置 (不是直接用nextTrackSamplePos)
            // 這部分比較複雜，我們先簡化為每個 track sample 點都嘗試畫一個立柱和連接到下一個的護欄
            if (i == 0 && side == -1) {  // 重置第一個立柱的累計距離參考點（只在開始處理第一側的第一個點時）
                prevPostBasePosForColor = postBasePos;
            } else if (i == 0 && side == 1) {
                prevPostBasePosForColor = postBasePos;  // 對於右側也重置
            }

            // 繪製立柱
            glColor3f(0.4f, 0.4f, 0.45f);  // 深灰色立柱
            glPushMatrix();
            glTranslatef(postCenterPos.x, postCenterPos.y + BARRIER_POST_HEIGHT / 2.0f, postCenterPos.z);
            // 立柱可以簡單地與世界軸對齊，或者與賽道法線對齊
            // 若要與賽道法線對齊 (使 POST_XZ_THICKNESS 的一個維度沿法線)
            // float angle = atan2f(segmentNormal.x, segmentNormal.z) * 180.0f / M_PI;
            // glRotatef(angle, 0, 1, 0);
            glScalef(POST_XZ_THICKNESS, BARRIER_POST_HEIGHT, POST_XZ_THICKNESS);
            glutSolidCube(1.0f);
            glPopMatrix();

            postCounterForReflector++;

            // --- 繪製反光片 (每隔數個立柱) ---
            if (postCounterForReflector % 4 == 0) {  // 每4個立柱一個反光片
                glPushMatrix();
                // 反光片應在立柱面向賽道的一側
                Vec3 reflectorPos =
                    postCenterPos - segmentNormal * side * (POST_XZ_THICKNESS / 2.0f + 0.01f);  // 貼在柱子表面
                reflectorPos.y += REFLECTOR_HEIGHT_OFFSET;

                glTranslatef(reflectorPos.x, reflectorPos.y, reflectorPos.z);
                // 使反光片面向賽道 (與 segmentNormal 方向相反)
                float reflectorAngle = atan2f(-segmentNormal.x * side, -segmentNormal.z * side) * 180.0f / M_PI;
                glRotatef(reflectorAngle, 0, 1, 0);

                glColor3f(1.0f, 0.8f, 0.2f);  // 黃色反光片
                glBegin(GL_QUADS);
                glVertex3f(-REFLECTOR_SIZE / 2, -REFLECTOR_SIZE / 4, 0);
                glVertex3f(REFLECTOR_SIZE / 2, -REFLECTOR_SIZE / 4, 0);
                glVertex3f(REFLECTOR_SIZE / 2, REFLECTOR_SIZE / 4, 0);
                glVertex3f(-REFLECTOR_SIZE / 2, REFLECTOR_SIZE / 4, 0);
                glEnd();
                glPopMatrix();
            }

            // 繪製連接到下一個 track sample 點的護欄
            if (trackSamples.size() > 1) {                  // 確保至少有兩個點
                Vec3 nextPostBasePos = nextTrackSamplePos;  // 下一個立柱的基座
                Vec3 nextPostCenterActual =
                    nextPostBasePos + segmentNormal * side * (SAND_HALF_W + BARRIER_OFFSET_FROM_SAND);
                nextPostCenterActual.y = nextPostBasePos.y;

                // 下層護欄 (銀灰色金屬質感)
                glColor3f(0.65f, 0.65f, 0.70f);  // 金屬灰色
                Vec3 railStartLower = postCenterPos;
                railStartLower.y += LOWER_RAIL_CENTER_Y_OFFSET;
                Vec3 railEndLower = nextPostCenterActual;
                railEndLower.y += LOWER_RAIL_CENTER_Y_OFFSET;
                drawRail(railStartLower, railEndLower, RAIL_BEAM_HEIGHT, RAIL_BEAM_DEPTH);

                // 上層護欄 (紅白相間警示色)
                // 使用從 prevPostBasePosForColor 到 postBasePos 的距離來決定顏色
                accumulatedDistanceOnSide += length(postBasePos - prevPostBasePosForColor);
                prevPostBasePosForColor = postBasePos;  // 更新參考點

                int colorSection = (int)(accumulatedDistanceOnSide / (POST_INTERVAL * 1.5f)) %
                                   2;  // 調整 POST_INTERVAL * X 來改變條紋寬度
                if (colorSection == 0) {
                    glColor3f(0.9f, 0.15f, 0.15f);  // 鮮紅色
                } else {
                    glColor3f(0.95f, 0.95f, 0.95f);  // 亮白色
                }
                Vec3 railStartUpper = postCenterPos;
                railStartUpper.y += UPPER_RAIL_CENTER_Y_OFFSET;
                Vec3 railEndUpper = nextPostCenterActual;
                railEndUpper.y += UPPER_RAIL_CENTER_Y_OFFSET;
                drawRail(railStartUpper, railEndUpper, RAIL_BEAM_HEIGHT, RAIL_BEAM_DEPTH);
            }
        }
    }
    glEnable(GL_LIGHTING);  // 如果場景中其他部分需要，則重新啟用光照
}
void drawRail(const Vec3& start, const Vec3& end, float beam_height, float beam_depth) {
    GLfloat current_color[4];
    glGetFloatv(GL_CURRENT_COLOR, current_color);  // 獲取外部設定的基礎顏色

    Vec3 dir = normalize(end - start);
    if (length(dir) < 1e-6f) return;  // 如果起點和終點太近，則不繪製

    Vec3 world_up = {0.0f, 1.0f, 0.0f};
    Vec3 rail_side_vector = normalize(cross(dir, world_up));

    // 如果 dir 與 world_up 平行 (例如垂直的柱子，雖然這裡是用於橫樑)
    // 選擇一個備用的 "up" 向量來計算 rail_side_vector
    if (length(rail_side_vector) < 0.1f) {
        Vec3 alternative_up = {1.0f, 0.0f, 0.0f};
        if (abs(dir.x) > 0.9f) alternative_up = {0.0f, 0.0f, 1.0f};  // 如果dir主要沿X軸，則用Z軸作爲alternative_up
        rail_side_vector = normalize(cross(dir, alternative_up));
    }
    if (length(rail_side_vector) < 1e-6f) rail_side_vector = {1.0f, 0.0f, 0.0f};  // 最後的備用

    Vec3 rail_up_vector = normalize(cross(rail_side_vector, dir));
    if (length(rail_up_vector) < 1e-6f) rail_up_vector = {0.0f, 1.0f, 0.0f};

    float h_half = beam_height / 2.0f;
    float d_half = beam_depth / 2.0f;

    // 計算8個頂點
    Vec3 s_bottom_left = start - rail_side_vector * d_half - rail_up_vector * h_half;
    Vec3 s_bottom_right = start + rail_side_vector * d_half - rail_up_vector * h_half;
    Vec3 s_top_right = start + rail_side_vector * d_half + rail_up_vector * h_half;
    Vec3 s_top_left = start - rail_side_vector * d_half + rail_up_vector * h_half;

    Vec3 e_bottom_left = end - rail_side_vector * d_half - rail_up_vector * h_half;
    Vec3 e_bottom_right = end + rail_side_vector * d_half - rail_up_vector * h_half;
    Vec3 e_top_right = end + rail_side_vector * d_half + rail_up_vector * h_half;
    Vec3 e_top_left = end - rail_side_vector * d_half + rail_up_vector * h_half;

    glBegin(GL_QUADS);

    // 頂面 (稍亮)
    glColor3f(min(1.0f, current_color[0] * 1.15f), min(1.0f, current_color[1] * 1.15f),
              min(1.0f, current_color[2] * 1.15f));
    glVertex3fv(&s_top_left.x);
    glVertex3fv(&s_top_right.x);
    glVertex3fv(&e_top_right.x);
    glVertex3fv(&e_top_left.x);

    // 底面 (稍暗)
    glColor3f(current_color[0] * 0.75f, current_color[1] * 0.75f, current_color[2] * 0.75f);
    glVertex3fv(&s_bottom_left.x);
    glVertex3fv(&e_bottom_left.x);
    glVertex3fv(&e_bottom_right.x);
    glVertex3fv(&s_bottom_right.x);

    // 正面 (+rail_side_vector 方向)
    glColor3fv(current_color);
    glVertex3fv(&s_bottom_right.x);
    glVertex3fv(&e_bottom_right.x);
    glVertex3fv(&e_top_right.x);
    glVertex3fv(&s_top_right.x);

    // 背面 (-rail_side_vector 方向)
    glColor3f(current_color[0] * 0.9f, current_color[1] * 0.9f, current_color[2] * 0.9f);  // 背面稍暗一些
    glVertex3fv(&s_bottom_left.x);
    glVertex3fv(&s_top_left.x);
    glVertex3fv(&e_top_left.x);
    glVertex3fv(&e_bottom_left.x);

    // 起始端面
    glColor3f(current_color[0] * 0.85f, current_color[1] * 0.85f, current_color[2] * 0.85f);
    glVertex3fv(&s_bottom_left.x);
    glVertex3fv(&s_bottom_right.x);
    glVertex3fv(&s_top_right.x);
    glVertex3fv(&s_top_left.x);

    // 結束端面
    glVertex3fv(&e_bottom_left.x);
    glVertex3fv(&e_top_left.x);
    glVertex3fv(&e_top_right.x);
    glVertex3fv(&e_bottom_right.x);

    glEnd();

    glColor3fv(current_color);  // 恢復外部顏色，以防影響後續繪製
}

// ---------- 中央斷續白線 ----------
void drawLaneMarkings() {
    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);               // 白色
    const float LINE_HALF_W = 0.12f;  // 線寬 0.24u
    const int PATTERN = 6;            // 取樣點間隔；越小越密

    glBegin(GL_QUADS);
    for (size_t i = 0; i < trackSamples.size(); ++i) {
        // 交錯顯示 dash：偶數段畫、奇數段留空
        if ((i / PATTERN) & 1) continue;

        const Vec3& p = trackSamples[i];
        const Vec3& q = trackSamples[(i + 1) % trackSamples.size()];

        Vec3 dir = normalize(q - p);
        Vec3 n = {-dir.z, 0, dir.x};   // 對路面垂直
        Vec3 a = p + n * LINE_HALF_W;  // 四個角點
        Vec3 b = p - n * LINE_HALF_W;
        Vec3 c = q - n * LINE_HALF_W;
        Vec3 d = q + n * LINE_HALF_W;

        glVertex3f(a.x, a.y + 0.015f, a.z);  // 微抬高避免 Z‑fight
        glVertex3f(b.x, b.y + 0.015f, b.z);
        glVertex3f(c.x, c.y + 0.015f, c.z);
        glVertex3f(d.x, d.y + 0.015f, d.z);
    }
    glEnd();
}
void drawEnvironmentModels() {
    // ok
    glPushMatrix();
    glTranslatef(-2500.0f, -5.0f, -200.0f);  // 將城市模型放置在世界原點 (之後可調整)
    glScalef(0.036f, 0.036f, 0.036f);        // !!! 這是一個猜測值，你需要根據模型實際大小調整 !!!
    glRotatef(45.0f, 0.0f, 1.0f, 0.0f);      // 例如，繞Y軸旋轉90度
    drawMultiMaterialModel(CasteliaCity);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(3000.0f, -20.0f, 0.0f);
    glScalef(190.0f, 190.0f, 190.0f);
    glRotatef(55.0f, 0.0f, 1.0f, 0.0f);
    drawMultiMaterialModel(Cliff);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(-3000.0f, -20.0f, 1000.0f);
    glScalef(190.0f, 190.0f, 190.0f);
    glRotatef(55.0f, 0.0f, 1.0f, 0.0f);
    drawMultiMaterialModel(Cliff);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(800.0f, -5.0f, 2450.0f);
    glScalef(0.026f, 0.026f, 0.026f);
    glRotatef(140.0f, 0.0f, 1.0f, 0.0f);
    drawMultiMaterialModel(TropicalIslands);
    glPopMatrix();
}

void makeShadowMatrix() {
    const GLfloat dot = PLANE[0] * LIGHT[0] + PLANE[1] * LIGHT[1] + PLANE[2] * LIGHT[2] + PLANE[3] * LIGHT[3];
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            // OpenGL 的矩陣是 column-major，所以索引是 col * 4 + row
            int index = col * 4 + row;
            GLfloat val = -LIGHT[row] * PLANE[col];
            if (row == col) {
                val += dot;
            }
            SHADOW[index] = val;
        }
    }
}

void drawCar3D() {
    auto drawShadow = [&](const Car& c) {
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_POLYGON_BIT);
        glDisable(GL_CULL_FACE);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);     // 讓影子不會寫入深度緩衝區
        glDisable(GL_DEPTH_TEST);  // 繪製影子時不進行深度測試，確保能畫在地面上
        glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
        // glColor4f(0.0f, 1.0f, 0.0f, 1.0f);  // 測試用：改為不透明亮綠色
        glPushMatrix();
        // 1) 套上正確的影子投影矩陣 (在 makeShadowMatrix 已修正)
        glMultMatrixf(SHADOW);
        // 2) 進行車子的模型變換
        glTranslatef(c.x, c.y, c.z);
        glRotatef(c.heading, 0, 1, 0);
        glScalef(0.08f, 0.08f, 0.08f);
        glRotatef(90.0f, 1, 0, 0);
        glRotatef(180.0f, 0, 1, 0);
        drawMultiMaterialModel(mustangModel);
        glPopMatrix();
        glDepthMask(GL_TRUE);  // 恢復深度緩衝區的寫入
        glPopAttrib();         // 還原所有屬性
    };
    // --- 繪製影子 ---
    drawShadow(car);
    if (gameMode != MODE_SOLO) {
        drawShadow(car2);
    }
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat light_pos[] = {0.5f, 1.0f, 0.5f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glEnable(GL_COLOR_MATERIAL);

    // --- 繪製 P1 車輛 ---
    glPushMatrix();
    glTranslatef(car.x, car.y, car.z);
    glRotatef(car.heading, 0, 1, 0);
    glScalef(0.08f, 0.08f, 0.08f);
    glTranslated(0.0, 0.0, 0.0);
    glRotatef(90.0f, 1, 0, 0);
    glRotatef(180.0f, 0, 1, 0);
    glColor3f(0.2f, 0.2f, 1);
    drawMultiMaterialModel(mustangModel);
    glPopMatrix();
    // --- 繪製 P2 車輛 ---
    if (gameMode != MODE_SOLO) {
        glPushMatrix();
        glTranslatef(car2.x, car2.y, car2.z);
        glRotatef(car2.heading, 0, 1, 0);
        glScalef(0.08f, 0.08f, 0.08f);
        glTranslated(0.0, 0.0, 0.0);
        glRotatef(90.0f, 1, 0, 0);
        glRotatef(180.0f, 0, 1, 0);
        glColor3f(1, 0.2f, 0.2f);
        drawMultiMaterialModel(mustangModel);
        glPopMatrix();
    }

    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
}
// (可選) 輔助函數：用方塊堆疊模擬圓柱體
void drawApproxCylinder(float radius, float height, int segments) {
    glPushMatrix();
    for (int i = 0; i < segments; ++i) {
        glPushMatrix();
        glRotatef((float)i * (360.0f / segments), 0, 1, 0);  // 繞Y軸旋轉
        glTranslatef(radius * 0.7f, 0, 0);                   // 稍微向外平移以形成圓周 (0.7是個近似值)

        // 計算方塊的尺寸使其看起來像圓柱的一部分
        float segmentWidth = 2.0f * M_PI * radius / segments * 1.2f;  // 段寬度，1.2f 是調整係數
        float segmentDepth = radius * 0.5f;                           // 段深度

        glScalef(segmentWidth, height, segmentDepth);
        glutSolidCube(1.0);
        glPopMatrix();
    }
    // 添加頂部和底部的蓋子 (可選，用扁平的圓盤或多邊形)
    // 為了簡單起見，這裡省略蓋子
    glPopMatrix();
}

// 修改後的 drawGate 函數
void drawGate(const Vec3& pos, float heading, bool isStartGate = true) {
    // --- 拱門尺寸參數 ---
    const float PILLAR_RADIUS = 0.6f;            // 立柱半徑 (如果用圓柱) 或 XZ厚度的一半 (如果用方柱)
    const float PILLAR_BASE_HEIGHT = 0.3f;       // 立柱底座高度
    const float PILLAR_BASE_RADIUS_MULT = 1.5f;  // 立柱底座半徑相對於立柱半徑的倍數
    const float PILLAR_MAIN_HEIGHT = 7.0f;       // 立柱主體高度
    const float PILLAR_CAP_HEIGHT = 0.4f;        // 立柱頂蓋高度
    const float PILLAR_CAP_RADIUS_MULT = 1.2f;   // 立柱頂蓋半徑倍數

    const float BEAM_SPAN = ROAD_HALF_W * 2.5f;  // 橫樑總跨度 (比路寬大一些)
    const float BEAM_HEIGHT = 1.2f;              // 橫樑的垂直厚度
    const float BEAM_DEPTH = 0.8f;               // 橫樑的深度 (Z方向，如果拱門面向X)

    const float SIGN_BOARD_WIDTH = BEAM_SPAN * 0.8f;     // 標誌板寬度
    const float SIGN_BOARD_HEIGHT = BEAM_HEIGHT * 1.5f;  // 標誌板高度
    const float SIGN_BOARD_DEPTH = 0.1f;                 // 標誌板厚度
    const float SIGN_BOARD_Y_OFFSET = PILLAR_MAIN_HEIGHT + PILLAR_BASE_HEIGHT + PILLAR_CAP_HEIGHT + BEAM_HEIGHT / 2.0f +
                                      SIGN_BOARD_HEIGHT / 2.0f + 0.2f;  // 標誌板Y軸偏移

    glPushMatrix();
    glTranslatef(pos.x, pos.y, pos.z);  // 平移到拱門的基準點 (通常是地面中心)
    glRotatef(heading, 0, 1, 0);        // 使拱門面向正確的方向

    // --- 繪製兩個立柱 ---
    for (int i = -1; i <= 1; i += 2) {  // i = -1 (左立柱), i = 1 (右立柱)
        glPushMatrix();
        // 將立柱平移到正確的橫向位置
        glTranslatef(i * (BEAM_SPAN / 2.0f - PILLAR_RADIUS * 0.5f), 0, 0);

        // 1. 立柱底座 (更寬、更矮的方塊或圓柱)
        glColor3f(0.3f, 0.3f, 0.35f);  // 暗灰色底座
        glPushMatrix();
        glTranslatef(0, PILLAR_BASE_HEIGHT / 2.0f, 0);
        glScalef(PILLAR_RADIUS * PILLAR_BASE_RADIUS_MULT * 2.0f, PILLAR_BASE_HEIGHT,
                 PILLAR_RADIUS * PILLAR_BASE_RADIUS_MULT * 2.0f);
        glutSolidCube(1.0);
        glPopMatrix();

        // 2. 立柱主體 (可以使用 glutSolidCylinder 或 drawApproxCylinder)
        glColor3f(0.5f, 0.5f, 0.55f);  // 中灰色立柱主體
        glPushMatrix();
        glTranslatef(0, PILLAR_BASE_HEIGHT + PILLAR_MAIN_HEIGHT / 2.0f, 0);
        // 如果使用 GLU 圓柱:
        // GLUquadric* quad = gluNewQuadric();
        // gluCylinder(quad, PILLAR_RADIUS, PILLAR_RADIUS, PILLAR_MAIN_HEIGHT, 20, 20);
        // gluDeleteQuadric(quad);
        // 否則，使用方塊模擬或 drawApproxCylinder:
        glScalef(PILLAR_RADIUS * 1.8f, PILLAR_MAIN_HEIGHT, PILLAR_RADIUS * 1.8f);  // 方形立柱
        glutSolidCube(1.0);
        glPopMatrix();

        // 3. 立柱頂蓋 (略寬於主體)
        glColor3f(0.4f, 0.4f, 0.45f);  // 頂蓋顏色
        glPushMatrix();
        glTranslatef(0, PILLAR_BASE_HEIGHT + PILLAR_MAIN_HEIGHT + PILLAR_CAP_HEIGHT / 2.0f, 0);
        glScalef(PILLAR_RADIUS * PILLAR_CAP_RADIUS_MULT * 2.0f, PILLAR_CAP_HEIGHT,
                 PILLAR_RADIUS * PILLAR_CAP_RADIUS_MULT * 2.0f);
        glutSolidCube(1.0);
        glPopMatrix();

        glPopMatrix();  // 完成此立柱的繪製
    }

    // --- 繪製橫樑 ---
    // 橫樑可以設計得更複雜，例如桁架結構，但這裡用一個實心長方體並在其上放置標誌板
    float beamBaseY = PILLAR_BASE_HEIGHT + PILLAR_MAIN_HEIGHT + PILLAR_CAP_HEIGHT;
    glColor3f(0.6f, 0.1f, 0.1f);  // 紅色橫樑
    glPushMatrix();
    glTranslatef(0, beamBaseY + BEAM_HEIGHT / 2.0f, 0);
    glScalef(BEAM_SPAN, BEAM_HEIGHT, BEAM_DEPTH);
    glutSolidCube(1.0);
    glPopMatrix();

    // --- 繪製標誌板 (START/FINISH) ---
    glPushMatrix();
    glTranslatef(0, SIGN_BOARD_Y_OFFSET, BEAM_DEPTH / 2.0f + SIGN_BOARD_DEPTH / 2.0f + 0.05f);  // 稍微向前偏移

    // 標誌板背景
    if (isStartGate) {
        glColor3f(0.1f, 0.6f, 0.1f);  // 起點用綠色背景
    } else {
        glColor3f(0.8f, 0.8f, 0.2f);  // 終點用黃色/棋盤格色背景 (簡化為黃色)
    }
    glPushMatrix();
    glScalef(SIGN_BOARD_WIDTH, SIGN_BOARD_HEIGHT, SIGN_BOARD_DEPTH);
    glutSolidCube(1.0);
    glPopMatrix();

    // 標誌板文字 (這部分在3D中用 glutBitmapCharacter 比較困難，通常會用紋理)
    // 為了簡單，我們可以在2D HUD中疊加文字，或者嘗試在3D中用線條繪製簡化文字
    // 這裡先不繪製3D文字，因為效果和控制比較麻煩
    // 如果要嘗試，可以參考 glutStrokeCharacter，但它繪製的是線框字
    // 例如:
    // glColor3f(1.0f, 1.0f, 1.0f); // 白色文字
    // glPushMatrix();
    // glTranslatef(-SIGN_BOARD_WIDTH * 0.3f, -SIGN_BOARD_HEIGHT*0.2f, SIGN_BOARD_DEPTH/2.0f + 0.01f); // 調整文字位置
    // glScalef(0.005f, 0.005f, 0.005f); // 調整文字大小
    // glLineWidth(2.0f);
    // const char* text = isStartGate ? "START" : "FINISH";
    // for (const char* p = text; *p; ++p) {
    //     glutStrokeCharacter(GLUT_STROKE_ROMAN, *p);
    // }
    // glPopMatrix();

    // 可選：添加棋盤格圖案到終點線的標誌板上 (用小方塊繪製)
    if (!isStartGate) {
        int numChecksX = 10;
        int numChecksY = 4;
        float checkSizeX = SIGN_BOARD_WIDTH / numChecksX;
        float checkSizeY = SIGN_BOARD_HEIGHT / numChecksY;
        for (int cy = 0; cy < numChecksY; ++cy) {
            for (int cx = 0; cx < numChecksX; ++cx) {
                if ((cx + cy) % 2 == 0) {
                    glColor3f(0.05f, 0.05f, 0.05f);  // 黑色格子
                } else {
                    glColor3f(0.95f, 0.95f, 0.95f);  // 白色格子
                }
                glPushMatrix();
                glTranslatef(-SIGN_BOARD_WIDTH / 2.0f + (cx + 0.5f) * checkSizeX,
                             -SIGN_BOARD_HEIGHT / 2.0f + (cy + 0.5f) * checkSizeY,
                             SIGN_BOARD_DEPTH / 2.0f + 0.01f  // 略微凸出
                );
                glScalef(checkSizeX, checkSizeY, SIGN_BOARD_DEPTH * 0.5f);  // 格子也略有厚度
                glutSolidCube(1.0);
                glPopMatrix();
            }
        }
    }

    glPopMatrix();  // 完成標誌板的繪製

    glPopMatrix();          // 完成整個拱門的繪製
    glEnable(GL_LIGHTING);  // 如果之前禁用了，確保重新啟用
}

// ------------- Mini‑map (top‑down orthographic) -------------
void drawUIButton(const UIButton& b, bool hover, float uiScale) {
    /* -------- (1) 按鈕陰影效果 ---------- */
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.0f, 0.0f, 0.0f, 0.4f);
    glBegin(GL_QUADS);
    glVertex2i(b.x + 4, b.y - 4);
    glVertex2i(b.x + b.w + 4, b.y - 4);
    glVertex2i(b.x + b.w + 4, b.y + b.h - 4);
    glVertex2i(b.x + 4, b.y + b.h - 4);
    glEnd();

    /* -------- (2) 按鈕漸變背景 ---------- */
    glBegin(GL_QUADS);
    if (hover) {
        // 懸停時使用更亮、更藍的漸變色
        glColor4f(0.4f, 0.6f, 0.9f, 0.95f);  // 頂部亮藍
        glVertex2i(b.x, b.y + b.h);
        glVertex2i(b.x + b.w, b.y + b.h);
        glColor4f(0.2f, 0.4f, 0.8f, 0.95f);  // 底部深藍
        glVertex2i(b.x + b.w, b.y);
        glVertex2i(b.x, b.y);
    } else {
        // 非懸停時使用灰藍漸變色
        glColor4f(0.7f, 0.7f, 0.8f, 0.95f);  // 頂部淺灰藍
        glVertex2i(b.x, b.y + b.h);
        glVertex2i(b.x + b.w, b.y + b.h);
        glColor4f(0.5f, 0.5f, 0.6f, 0.95f);  // 底部深灰藍
        glVertex2i(b.x + b.w, b.y);
        glVertex2i(b.x, b.y);
    }
    glEnd();

    /* -------- (3) 內部亮邊 ---------- */
    glColor4f(1.0f, 1.0f, 1.0f, 0.7f);
    glLineWidth(1.0f);
    glBegin(GL_LINE_STRIP);
    glVertex2i(b.x + 1, b.y + b.h - 1);
    glVertex2i(b.x + 1, b.y + 1);
    glVertex2i(b.x + b.w - 1, b.y + 1);
    glEnd();

    /* -------- (4) 外部邊框 ---------- */
    glColor4f(0.3f, 0.3f, 0.5f, 0.8f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2i(b.x, b.y);
    glVertex2i(b.x + b.w, b.y);
    glVertex2i(b.x + b.w, b.y + b.h);
    glVertex2i(b.x, b.y + b.h);
    glEnd();

    /* -------- (5) 文字與陰影 ---------- */
    const int FONT_H = 18;                    // HELVETICA_18 的高度
    int txtY = b.y + (b.h - FONT_H) / 2 + 4;  // +4 讓 baseline 稍微居中

    // 計算文字寬度以實現居中
    int textWidth = 0;
    for (const char* p = b.text; *p; ++p) {
        textWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_18, *p);
    }
    int txtX = b.x + (b.w - textWidth) / 2;  // 文字居中顯示

    // 文字陰影
    glColor4f(0.1f, 0.1f, 0.3f, 0.7f);
    glRasterPos2i(txtX + 1, txtY - 1);
    for (const char* p = b.text; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);

    // 文字本身
    glColor3f(hover ? 1.0f : 0.0f, hover ? 1.0f : 0.0f, hover ? 1.0f : 0.0f);  // 懸停時為白色，否則為黑色
    glRasterPos2i(txtX, txtY);
    for (const char* p = b.text; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);

    glDisable(GL_BLEND);
}

// 繪製首頁選單
void drawMainMenu(int mouseX = -1, int mouseY = -1) {
    /* -- UI 尺寸等比例縮放 (以 720 為 100%) -- */
    float uiScale = winH / 720.0f;

    /* 重新計算並寫回按鈕資料 ── 讓 hit‑test 也正確 */
    btnSolo.w = int(UI_BTN_W0 * uiScale + 0.5f);
    btnSolo.h = int(UI_BTN_H0 * uiScale + 0.5f);
    btnSolo.x = int(UI_BTN_X0 * uiScale + 0.5f);

    btnPVP.w = btnSolo.w;
    btnPVP.h = btnSolo.h;
    btnPVP.x = btnSolo.x;

    btnAI.w = btnSolo.w;
    btnAI.h = btnSolo.h;
    btnAI.x = btnSolo.x;

    btnPreview.w = btnSolo.w;
    btnPreview.h = btnSolo.h;
    btnPreview.x = btnSolo.x;

    /* 垂直位置：視窗中央為基準 */
    int centerY = winH / 2;
    btnSolo.y = centerY + int(UI_BTN_GAP * 1.25f * uiScale);     // 上方按鈕
    btnPVP.y = centerY + int(UI_BTN_GAP * 0.25f * uiScale);      // 中間按鈕
    btnAI.y = centerY - int(UI_BTN_GAP * 0.75f * uiScale);       // 下方按鈕
    btnPreview.y = centerY - int(UI_BTN_GAP * 1.75f * uiScale);  // 放置在最下方

    int titleX = btnSolo.x + btnSolo.w / 2 - 80;  // 標題X位置調整為居中
    int titleY = btnSolo.y + btnSolo.h + int(UI_TITLE_OF * uiScale + 0.5f);

    // 背景貼圖
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texMenu);
    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2i(0, 0);
    glTexCoord2f(1, 0);
    glVertex2i(winW, 0);
    glTexCoord2f(1, 1);
    glVertex2i(winW, winH);
    glTexCoord2f(0, 1);
    glVertex2i(0, winH);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    /* ---------- 更新雲座標 ---------- */
    static int prev = glutGet(GLUT_ELAPSED_TIME);
    int now = glutGet(GLUT_ELAPSED_TIME);
    float dt = (now - prev) * 0.001f;
    prev = now;

    for (auto& c : clouds) {
        c.x += c.speed * dt;
        if (c.x - c.w > winW) c.x = -c.w;  // 從左邊回來
    }

    /* ---------- 繪製雲 ---------- */
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    /* 1) 柔邊漸層圓 ------------------------------------------- */
    auto softCircle = [&](float cx, float cy, float r, float cr, float cg, float cb) {
        const int SEG = 60;
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(cr, cg, cb, 0.9f);  // 中心亮
        glVertex2f(cx, cy);
        for (int i = 0; i <= SEG; ++i) {
            float ang = i * 2.0f * M_PI / SEG;
            float x = cx + cosf(ang) * r;
            float y = cy + sinf(ang) * r;
            glColor4f(cr, cg, cb, 0.0f);  // 外緣透明
            glVertex2f(x, y);
        }
        glEnd();
    };

    /* 2) 底部陰影橢圓 ----------------------------------------- */
    auto cloudShadow = [&](float cx, float cy, float rw, float rh) {
        const int SEG = 40;
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(0, 0, 0, 0.18f);
        glVertex2f(cx, cy);
        for (int i = 0; i <= SEG; ++i) {
            float ang = i * 2.0f * M_PI / SEG;
            float x = cx + cosf(ang) * rw;
            float y = cy + sinf(ang) * rh;
            glColor4f(0, 0, 0, 0.0f);
            glVertex2f(x, y);
        }
        glEnd();
    };

    /* 3) 繪製每朵雲 ------------------------------------------- */
    for (auto& c : clouds) {
        float cx = c.x + c.w * 0.5f;
        float cy = c.y + c.h * 0.5f;
        float r = c.h * 0.45f;

        /* 陰影：下偏 16 像素，扁橢圓 */
        cloudShadow(cx, cy - 16, r * 1.3f, r * 0.6f);

        /* 主體：8 顆圓帶隨機微位移 */
        struct P {
            float dx, dy, mul;
        };
        const P part[8] = {{-1.4f, 0.0f, 1.05f},  {-0.6f, 0.35f, 0.95f}, {0.6f, 0.35f, 0.95f}, {1.4f, 0.0f, 1.05f},
                           {-0.8f, 0.95f, 0.75f}, {0.0f, 1.05f, 0.70f},  {0.8f, 0.95f, 0.75f}, {0.0f, -0.25f, 0.80f}};

        for (auto p : part) {
            /* 為了不每幀都變，seed 用雲指標地址 */
            float jx = ((size_t)&c % 97) * 0.005f;  // ±0.5%
            float jy = ((size_t)&c % 83) * 0.005f;
            float ox = p.dx * r * (1.0f + jx);
            float oy = p.dy * r * (1.0f + jy);
            float rr = r * p.mul * (1.0f + jx * 0.5f);

            /* 顏色：上半亮白，下半淡灰藍 → 高度決定色調 */
            float t = (oy + r) / (2 * r);  // 0 底部 → 1 頂部
            float cr = 1.0f;
            float cg = 1.0f;
            float cb = 1.0f - 0.12f * (1 - t);  // 底部帶 12% 藍灰

            softCircle(cx + ox, cy + oy, rr, cr, cg, cb);
        }
    }

    glDisable(GL_BLEND);

    /* ---------- 更新 & 畫塵土 (右上飄) ---------- */
    for (auto& d : dust) {
        d.x += d.vx * dt;
        d.y += d.vy * dt;
        d.a -= dt * 0.08f;  // 緩慢淡出

        // 超出右邊或頂端，或完全透明 → 重生
        if (d.x > winW || d.y > winH || d.a <= 0.01f) {
            d.x = (rand() % winW) + 200.0f;  // 任意 X
            d.y = 50.0f;
            d.r = 2.0f + (rand() % 70) * 0.04f;
            float s = 30.0f + rand() % 50;
            d.vx = s * 0.707f;
            d.vy = s * 0.707f;
            d.a = 0.15f + (rand() & 1) * 0.1f;
        }
    }

    /* 畫塵土：沙黃色柔邊小圓 */
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const float dcR = 0.82f, dcG = 0.72f, dcB = 0.55f;
    for (const auto& d : dust) {
        const int SEG = 12;
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(dcR, dcG, dcB, d.a);
        glVertex2f(d.x, d.y);
        glColor4f(dcR, dcG, dcB, 0.0f);
        for (int i = 0; i <= SEG; ++i) {
            float ang = i * 2.0f * M_PI / SEG;
            glVertex2f(d.x + cosf(ang) * d.r, d.y + sinf(ang) * d.r);
        }
        glEnd();
    }
    glDisable(GL_BLEND);

    /* ---------- Leaderboard (右側) ---------- */
    /* -- 根據目前高度計算縮放因子 (以 720 為基準) -- */
    float scale = winH / 720.0f;

    /* 盒子尺寸與邊距 - 重新計算確保所有內容都有足夠空間 */
    int boxW = int(300 * scale + 0.5f);                // 進一步加寬以避免文字跑版
    int rowH = int(32 * scale + 0.5f);                 // 略微增加行高
    int boxH = TOP_N * rowH + int(70 * scale + 0.5f);  // 增加標題高度
    int marginX = int(50 * scale + 0.5f);              // 右邊留 50
    int marginY = int(150 * scale + 0.5f);             // 底端留 150

    int boxX = winW - boxW - marginX;
    int boxY = winH - boxH - marginY;

    // 首先繪製陰影效果增加深度感
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    glColor4f(0.0f, 0.0f, 0.0f, 0.3f);
    glVertex2i(boxX + 6, boxY - 6);
    glVertex2i(boxX + boxW + 6, boxY - 6);
    glVertex2i(boxX + boxW + 6, boxY + boxH - 6);
    glVertex2i(boxX + 6, boxY + boxH - 6);
    glEnd();

    // 主背景 - 使用更均勻的深藍色
    glBegin(GL_QUADS);
    glColor4f(0.08f, 0.1f, 0.22f, 0.9f);
    glVertex2i(boxX, boxY);
    glVertex2i(boxX + boxW, boxY);
    glVertex2i(boxX + boxW, boxY + boxH);
    glVertex2i(boxX, boxY + boxH);
    glEnd();

    // 使用双邊框效果增加立體感
    // 外邊框 - 深色
    glLineWidth(3.0f);
    glBegin(GL_LINE_LOOP);
    glColor4f(0.2f, 0.2f, 0.5f, 0.8f);
    glVertex2i(boxX, boxY);
    glVertex2i(boxX + boxW, boxY);
    glVertex2i(boxX + boxW, boxY + boxH);
    glVertex2i(boxX, boxY + boxH);
    glEnd();

    // 內邊框 - 亮色
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
    glColor4f(0.5f, 0.6f, 1.0f, 0.8f);
    glVertex2i(boxX + 2, boxY + 2);
    glVertex2i(boxX + boxW - 2, boxY + 2);
    glVertex2i(boxX + boxW - 2, boxY + boxH - 2);
    glVertex2i(boxX + 2, boxY + boxH - 2);
    glEnd();

    // 標題區域 - 使用更醒目的顏色和漸變
    int titleHeight = int(45 * scale + 0.5f);
    glBegin(GL_QUADS);
    glColor4f(0.15f, 0.2f, 0.5f, 0.95f);
    glVertex2i(boxX, boxY + boxH);
    glVertex2i(boxX + boxW, boxY + boxH);
    glColor4f(0.1f, 0.15f, 0.4f, 0.95f);
    glVertex2i(boxX + boxW, boxY + boxH - titleHeight);
    glVertex2i(boxX, boxY + boxH - titleHeight);
    glEnd();

    // 標題文字 - 確保居中
    void* titleFont = GLUT_BITMAP_HELVETICA_18;
    const char* lb = "LEADERBOARD";

    // 計算文字寬度以確保居中
    int titleWidth = 0;
    for (const char* p = lb; *p; ++p) {
        titleWidth += glutBitmapWidth(titleFont, *p);
    }

    int leaderTitleX = boxX + (boxW - titleWidth) / 2;
    int leaderTitleY = boxY + boxH - titleHeight / 2 - 5;

    // 標題文字陰影效果
    glColor3f(0.0f, 0.0f, 0.2f);
    glRasterPos2i(leaderTitleX + 1, leaderTitleY - 1);
    for (const char* p = lb; *p; ++p) {
        glutBitmapCharacter(titleFont, *p);
    }

    // 標題文字
    glColor3f(1.0f, 0.9f, 0.3f);  // 金黃色
    glRasterPos2i(leaderTitleX, leaderTitleY);
    for (const char* p = lb; *p; ++p) {
        glutBitmapCharacter(titleFont, *p);
    }

    // 標題分隔線
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor4f(0.4f, 0.5f, 1.0f, 0.8f);
    glVertex2i(boxX + 10, boxY + boxH - titleHeight);
    glVertex2i(boxX + boxW - 10, boxY + boxH - titleHeight);
    glEnd();

    // 定義列寬和位置
    const int rankColWidth = int(80 * scale);
    const int timeColWidth = boxW - rankColWidth;

    // 欄位標題
    void* headerFont = GLUT_BITMAP_HELVETICA_12;
    int headerY = boxY + boxH - titleHeight - 15;

    glColor3f(0.7f, 0.7f, 1.0f);

    // 排名欄標題
    const char* rankHeader = "RANK";
    int rankHeaderWidth = 0;
    for (const char* p = rankHeader; *p; ++p) {
        rankHeaderWidth += glutBitmapWidth(headerFont, *p);
    }
    int rankCenterX = boxX + rankColWidth / 2;
    glRasterPos2i(rankCenterX - rankHeaderWidth / 2, headerY);
    for (const char* p = rankHeader; *p; ++p) {
        glutBitmapCharacter(headerFont, *p);
    }

    // 時間欄標題
    const char* timeHeader = "TIME";
    int timeHeaderWidth = 0;
    for (const char* p = timeHeader; *p; ++p) {
        timeHeaderWidth += glutBitmapWidth(headerFont, *p);
    }
    int timeCenterX = boxX + rankColWidth + timeColWidth / 2;
    glRasterPos2i(timeCenterX - timeHeaderWidth / 2, headerY);
    for (const char* p = timeHeader; *p; ++p) {
        glutBitmapCharacter(headerFont, *p);
    }

    // 列表內容 - 重新設計確保對齊
    void* contentFont = GLUT_BITMAP_HELVETICA_18;

    // 繪製分隔線
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glColor4f(0.3f, 0.4f, 0.8f, 0.5f);
    glVertex2i(boxX + 10, headerY - 5);
    glVertex2i(boxX + boxW - 10, headerY - 5);
    glEnd();

    // 繪製垂直分隔線
    glBegin(GL_LINES);
    glColor4f(0.3f, 0.4f, 0.8f, 0.3f);
    glVertex2i(boxX + rankColWidth, headerY + 15);
    glVertex2i(boxX + rankColWidth, boxY + 20);
    glEnd();

    for (int i = 0; i < (int)scores.size() && i < TOP_N; ++i) {
        int rowY = headerY - 32 - i * rowH;

        // 行背景 - 交替顏色
        if (i % 2 == 0) {
            glBegin(GL_QUADS);
            glColor4f(0.15f, 0.15f, 0.3f, 0.5f);
            glVertex2i(boxX + 5, rowY + 15);
            glVertex2i(boxX + boxW - 5, rowY + 15);
            glVertex2i(boxX + boxW - 5, rowY - rowH + 15);
            glVertex2i(boxX + 5, rowY - rowH + 15);
            glEnd();
        }

        // 設置排名數字和時間
        char rankStr[8];
        char timeStr[16];
        sprintf(rankStr, "%d", i + 1);
        int total = (int)scores[i];
        sprintf(timeStr, "%02d:%05.2f", total / 60, fmod(scores[i], 60.0f));

        // 計算排名文字寬度居中顯示
        int rankWidth = 0;
        for (char* p = rankStr; *p; ++p) {
            rankWidth += glutBitmapWidth(contentFont, *p);
        }

        // 排名顏色
        if (i == 0) {
            glColor3f(1.0f, 0.84f, 0.0f);  // 金色
        } else if (i == 1) {
            glColor3f(0.75f, 0.75f, 0.75f);  // 銀色
        } else if (i == 2) {
            glColor3f(0.8f, 0.5f, 0.2f);  // 銅色
        } else {
            glColor3f(0.7f, 0.7f, 0.7f);  // 其他名次
        }

        // 排名文字
        glRasterPos2i(rankCenterX - rankWidth / 2, rowY);
        for (char* p = rankStr; *p; ++p) {
            glutBitmapCharacter(contentFont, *p);
        }

        // 前三名添加獎牌
        if (i < 3) {
            float medalSize = 10 * scale;
            // 獎牌位置 - 在排名數字左側
            float medalX = rankCenterX + rankWidth / 2 - 25;
            float medalY = rowY + 7;

            // 獎牌外圈
            glBegin(GL_TRIANGLE_FAN);
            for (int j = 0; j <= 360; j += 30) {
                float angle = j * M_PI / 180.0f;
                glVertex2f(medalX + cosf(angle) * medalSize, medalY + sinf(angle) * medalSize * 0.9f);
            }
            glEnd();

            // 獎牌內圈 - 稍微不同的顏色
            if (i == 0) glColor3f(0.9f, 0.7f, 0.0f);     // 金色內部
            if (i == 1) glColor3f(0.65f, 0.65f, 0.65f);  // 銀色內部
            if (i == 2) glColor3f(0.7f, 0.4f, 0.1f);     // 銅色內部

            glBegin(GL_TRIANGLE_FAN);
            for (int j = 0; j <= 360; j += 30) {
                float angle = j * M_PI / 180.0f;
                glVertex2f(medalX + cosf(angle) * (medalSize - 2), medalY + sinf(angle) * (medalSize - 2) * 0.9f);
            }
            glEnd();
        }

        // 計算時間文字寬度居中顯示
        int timeWidth = 0;
        for (char* p = timeStr; *p; ++p) {
            timeWidth += glutBitmapWidth(contentFont, *p);
        }

        // 時間文字
        glColor3f(1.0f, 1.0f, 1.0f);
        glRasterPos2i(timeCenterX - timeWidth / 2, rowY);
        for (char* p = timeStr; *p; ++p) {
            glutBitmapCharacter(contentFont, *p);
        }
    }

    // 空記錄顯示
    for (int i = scores.size(); i < TOP_N; i++) {
        int rowY = headerY - 20 - i * rowH;

        // 交替背景
        if (i % 2 == 0) {
            glBegin(GL_QUADS);
            glColor4f(0.15f, 0.15f, 0.3f, 0.5f);
            glVertex2i(boxX + 5, rowY + 15);
            glVertex2i(boxX + boxW - 5, rowY + 15);
            glVertex2i(boxX + boxW - 5, rowY - rowH + 15);
            glVertex2i(boxX + 5, rowY - rowH + 15);
            glEnd();
        }

        // 排名
        char rankStr[8];
        sprintf(rankStr, "%d", i + 1);
        int rankWidth = 0;
        for (char* p = rankStr; *p; ++p) {
            rankWidth += glutBitmapWidth(contentFont, *p);
        }

        glColor3f(0.5f, 0.5f, 0.5f);
        glRasterPos2i(rankCenterX - rankWidth / 2, rowY);
        for (char* p = rankStr; *p; ++p) {
            glutBitmapCharacter(contentFont, *p);
        }

        // 時間占位符
        const char* placeholder = "--:--.--";
        int placeholderWidth = 0;
        for (const char* p = placeholder; *p; ++p) {
            placeholderWidth += glutBitmapWidth(contentFont, *p);
        }

        glRasterPos2i(timeCenterX - placeholderWidth / 2, rowY);
        for (const char* p = placeholder; *p; ++p) {
            glutBitmapCharacter(contentFont, *p);
        }
    }

    bool hovS = (mouseX >= btnSolo.x && mouseX <= btnSolo.x + btnSolo.w && mouseY >= btnSolo.y &&
                 mouseY <= btnSolo.y + btnSolo.h);
    bool hovP =
        (mouseX >= btnPVP.x && mouseX <= btnPVP.x + btnPVP.w && mouseY >= btnPVP.y && mouseY <= btnPVP.y + btnPVP.h);
    bool hovA = (mouseX >= btnAI.x && mouseX <= btnAI.x + btnAI.w && mouseY >= btnAI.y && mouseY <= btnAI.y + btnAI.h);
    bool hovPreview = (mouseX >= btnPreview.x && mouseX <= btnPreview.x + btnPreview.w && mouseY >= btnPreview.y &&
                       mouseY <= btnPreview.y + btnPreview.h);  // 新按鈕的滑鼠懸停檢測

    drawUIButton(btnSolo, hovS, uiScale);
    drawUIButton(btnPVP, hovP, uiScale);
    drawUIButton(btnAI, hovA, uiScale);
    drawUIButton(btnPreview, hovPreview, uiScale);
    // 添加裝飾線
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glColor4f(0.6f, 0.7f, 1.0f, 0.6f);
    glVertex2i(btnSolo.x - 10, centerY);
    glVertex2i(btnSolo.x + btnSolo.w + 10, centerY);
    glEnd();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();  // MODEL
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

/* -------- 倒數大字顯示 -------------------------------- */
void drawCountdown(bool isLeftSide = false) {
    if (!countdownActive) return;

    /* 1. 目前要顯示的字串 ------------------------------ */
    int num = int(countdownLeft);             // 3,2,1,0
    const char* txt = (countdownLeft > 1.0f)  // >1 秒 ⇒ 3/2/1
                          ? (num == 3   ? "3"
                             : num == 2 ? "2"
                                        : "1")
                          : "GO!";  // 1 秒內 ⇒ GO!

    /* 2. 前置: 進 2D 投影、關掉深度 ---------------------- */
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    /* 3. 字串不透明度：最後 0.2 秒淡出 -------------------- */
    float alpha = 1.0f;
    if (countdownLeft < 0.2f)          // 0.0 ~ 0.2 秒
        alpha = countdownLeft / 0.2f;  // 線性淡出

    glColor4f(1, 1, 0, alpha);  // 黃色

    /* 4. 使用 GLUT_STROKE_ROMAN，算出原始寬度 ------------ */
    void* font = GLUT_STROKE_ROMAN;

    float w0 = 0.0f;
    for (const char* p = txt; *p; ++p) w0 += glutStrokeWidth(font, *p);  // 原始字寬 (單位: 1)

    const float H0 = 119.05f;  // GLUT_STROKE_ROMAN 的字高

    /* 5. 想要的螢幕高度 (px) ---------------------------- */
    float targetH = winH * 0.25f;  // 25% 視窗高
    float scale = targetH / H0;    // 轉成縮放因子

    /* 6. 左下角座標 (px) – 置中，且在螢幕 65% 高度處 ------ */
    float drawW = w0 * scale;

    // 調整X座標以適應分屏模式
    float posX;
    if (gameMode == MODE_PVP || gameMode == MODE_AI) {
        // 分屏模式下，倒數計時居中於各自的視窗
        if (isLeftSide) {
            posX = (winW / 4) - (drawW / 2);  // 左側視窗中央
        } else {
            posX = (winW * 3 / 4) - (drawW / 2);  // 右側視窗中央
        }
    } else {
        // 單視窗模式下，倒數計時居中於整個視窗
        posX = (winW - drawW) * 0.5f;
    }

    float posY = winH * 0.65f - targetH * 0.5f;

    /* 7. 實際繪製 -------------------------------------- */
    glTranslatef(posX, posY, 0);
    glScalef(scale, scale, 1);

    for (const char* p = txt; *p; ++p) glutStrokeCharacter(font, *p);

    /* 8. 還原狀態 -------------------------------------- */
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}
void drawTrackPreviewUI() {
    // ---- 2D 畫面座標 ----
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);  // 左下 (0,0)

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const char* text1 = "Drag Mouse to Pan Camera";
    const char* text2 = "Press ESC to Return to Menu";

    // 設定文字顏色
    glColor4f(1.0f, 1.0f, 1.0f, 0.8f);  // 白色半透明

    // 計算文字位置 (左上角)
    int x_pos = 20;
    int y_pos1 = winH - 30;
    int y_pos2 = winH - 55;

    glRasterPos2i(x_pos, y_pos1);
    for (const char* p = text1; *p; ++p) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
    }

    glRasterPos2i(x_pos, y_pos2);
    for (const char* p = text2; *p; ++p) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
    }

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // 還原
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}
void drawMiniMap(bool isLeftSide = false) {
    // 小地圖尺寸根據視窗大小動態調整
    const float miniMapRatio = 0.15f;             // 小地圖寬度佔視窗寬度的比例
    const int SIZE = (int)(winW * miniMapRatio);  // 動態計算大小
    const int MARGIN = (int)(winW * 0.01f);       // 邊距也動態調整

    // 根據是否為左側視窗來設置位置
    if (isLeftSide) {
        // 左側視窗的小地圖在左半部的右上角
        glViewport(winW / 2 - SIZE - MARGIN, winH - SIZE - MARGIN, SIZE, SIZE);
    } else {
        // 右側視窗的小地圖在右半部的右上角 (或全屏模式下)
        glViewport(winW - SIZE - MARGIN, winH - SIZE - MARGIN, SIZE, SIZE);
    }

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(mapMinX, mapMaxX, mapMinZ, mapMaxZ, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);

    // 啟用混合模式，使背景能夠半透明
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 背景方塊，更加透明
    glColor4f(0, 0, 0, 0.25f);  // 透明度從0.4降至0.25
    glBegin(GL_QUADS);
    glVertex2f(mapMinX, mapMinZ);
    glVertex2f(mapMaxX, mapMinZ);
    glVertex2f(mapMaxX, mapMaxZ);
    glVertex2f(mapMinX, mapMaxZ);
    glEnd();

    // 賽道線條
    glColor3f(1, 1, 1);
    glLineWidth(2);
    glBegin(GL_LINE_LOOP);  // 使用LINE_LOOP而不是LINE_STRIP來自動閉合
    for (auto& p : trackSamples) glVertex2f(p.x, p.z);
    glEnd();

    // 起點與終點標記
    glColor3f(0.0f, 1.0f, 0.0f);  // 綠色起點
    glPointSize(8);
    glBegin(GL_POINTS);
    glVertex2f(gateStartPos.x, gateStartPos.z);
    glEnd();

    glColor3f(1.0f, 0.0f, 0.0f);  // 紅色終點
    glPointSize(8);
    glBegin(GL_POINTS);
    glVertex2f(gateFinishPos.x, gateFinishPos.z);
    glEnd();

    // 小地圖點大小也根據視窗大小調整
    float pointSize = max(4.0f, SIZE / 40.0f);

    // 第一輛車(藍)點和朝向
    glPointSize(pointSize);
    glColor3f(0.2f, 0.2f, 1.0f);
    glBegin(GL_POINTS);
    glVertex2f(car.x, car.z);
    glEnd();

    float hdgLen = (mapMaxX - mapMinX) * 0.03f;
    float rad = car.heading * M_PI / 180;
    glBegin(GL_LINES);
    glVertex2f(car.x, car.z);
    glVertex2f(car.x + sinf(rad) * hdgLen, car.z + cosf(rad) * hdgLen);
    glEnd();

    // 第二輛車(紅)點和朝向
    glPointSize(pointSize);
    glColor3f(1.0f, 0.2f, 0.2f);
    glBegin(GL_POINTS);
    glVertex2f(car2.x, car2.z);
    glEnd();

    rad = car2.heading * M_PI / 180;
    glBegin(GL_LINES);
    glVertex2f(car2.x, car2.z);
    glVertex2f(car2.x + sinf(rad) * hdgLen, car2.z + cosf(rad) * hdgLen);
    glEnd();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // 恢復矩陣
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    // 恢復全視窗視口
    glViewport(0, 0, winW, winH);
}
void drawHUDTimer(bool isLeftSide) {
    // ------ 2D 畫面座標 ------
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);  // 左下 (0,0)

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 轉成 mm:ss.cc
    int total = (int)raceTime;
    int minutes = total / 60;
    int seconds = total % 60;
    int centis = (int)(fmod(raceTime, 1.0f) * 100);  // Corrected centisecond calculation

    char buf[16];
    sprintf(buf, "%02d:%02d.%02d", minutes, seconds, centis);

    // 計算文字寬度 (使用較大字體)
    int textWidth = 0;
    for (char* p = buf; *p; ++p) textWidth += glutBitmapWidth(GLUT_BITMAP_TIMES_ROMAN_24, *p);

    const int boxHeight = 40;  // Increased height for better padding
    const int boxPadding = 10;
    const int boxWidth = textWidth + 2 * boxPadding;

    // 計算位置：根據分屏模式決定置中位置
    int boxX;
    if (gameMode == MODE_PVP || gameMode == MODE_AI) {
        // 分屏模式：各自中上方
        boxX = isLeftSide ? (winW / 4) - (boxWidth / 2) : (winW * 3 / 4) - (boxWidth / 2);
    } else {
        // 單視窗模式：總寬度的中間
        boxX = (winW / 2) - (boxWidth / 2);
    }
    int boxY = winH - boxHeight - 15;  // Slightly lower position

    // 繪製背景框 (深色半透明)
    glColor4f(0.0f, 0.0f, 0.0f, 0.6f);  // Darker, more transparent background
    glBegin(GL_QUADS);
    glVertex2i(boxX, boxY);
    glVertex2i(boxX + boxWidth, boxY);
    glVertex2i(boxX + boxWidth, boxY + boxHeight);
    glVertex2i(boxX, boxY + boxHeight);
    glEnd();

    // 繪製邊框 (亮色)
    glColor4f(0.8f, 0.8f, 1.0f, 0.7f);  // Light blueish-white border
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2i(boxX, boxY);
    glVertex2i(boxX + boxWidth, boxY);
    glVertex2i(boxX + boxWidth, boxY + boxHeight);
    glVertex2i(boxX, boxY + boxHeight);
    glEnd();

    // 文字顏色和位置
    if (!timerRunning)
        glColor3f(1.0f, 0.3f, 0.3f);  // 停表後改成較亮的紅色
    else
        glColor3f(1.0f, 1.0f, 1.0f);  // 白色文字

    int textX = boxX + boxPadding;
    int textY = boxY + (boxHeight - 24) / 2 + 5;  // Vertically center TIMES_ROMAN_24 (approx height 24)

    glRasterPos2i(textX, textY);
    for (char* p = buf; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *p);  // Use larger font

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // 還原
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}
void drawBackground() {
    glBegin(GL_QUADS);

    // 天空頂部顏色 - 深藍紫色
    GLfloat skyTopColor[] = {0.2f, 0.15f, 0.4f};  // 深藍紫
    // 地平線顏色 - 暖橙色或淡紫色
    GLfloat horizonColor[] = {0.85f, 0.45f, 0.3f};  // 暖橙紅色
    // 您也可以嘗試這種地平線顏色，更偏紫一點：
    // GLfloat horizonColor[] = {0.6f, 0.4f, 0.7f}; // 淡紫色

    // --- 前方天空 ---
    glColor3fv(skyTopColor);
    glVertex3f(-7000, 2000, -7000);  // 左上
    glVertex3f(7000, 2000, -7000);   // 右上
    glColor3fv(horizonColor);
    glVertex3f(7000, -100, -7000);   // 右下 (地平線)
    glVertex3f(-7000, -100, -7000);  // 左下 (地平線)

    // --- 後方天空 ---
    glColor3fv(skyTopColor);
    glVertex3f(-7000, 2000, 7000);  // 左上
    glVertex3f(7000, 2000, 7000);   // 右上
    glColor3fv(horizonColor);
    glVertex3f(7000, -100, 7000);   // 右下 (地平線)
    glVertex3f(-7000, -100, 7000);  // 左下 (地平線)

    // --- 左側天空 ---
    glColor3fv(skyTopColor);
    glVertex3f(-7000, 2000, -7000);  // 後上
    glVertex3f(-7000, 2000, 7000);   // 前上
    glColor3fv(horizonColor);
    glVertex3f(-7000, -100, 7000);   // 前下 (地平線)
    glVertex3f(-7000, -100, -7000);  // 後下 (地平線)

    // --- 右側天空 ---
    glColor3fv(skyTopColor);
    glVertex3f(7000, 2000, -7000);  // 後上
    glVertex3f(7000, 2000, 7000);   // 前上
    glColor3fv(horizonColor);
    glVertex3f(7000, -100, 7000);   // 前下 (地平線)
    glVertex3f(7000, -100, -7000);  // 後下 (地平線)

    // --- 天空頂部 (直接使用天空頂部顏色) ---
    glColor3fv(skyTopColor);
    glVertex3f(-7000, 2000, -7000);
    glVertex3f(7000, 2000, -7000);
    glVertex3f(7000, 2000, 7000);
    glVertex3f(-7000, 2000, 7000);

    glEnd();
}
// 添加時速表顯示函數
void drawSpeedometer(float speed, bool isLeftSide) {
    // 轉換為時速
    float kmh = fabsf(speed) * 3.6f;  // 假設遊戲單位是m/s, 乘3.6轉為km/h

    // 2D 畫面座標設置
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 設定位置和大小
    int radius = 70;  // 儀表板半徑
    int centerX;
    if (gameMode == MODE_PVP || gameMode == MODE_AI) {
        centerX = isLeftSide ? radius + 30 : winW / 2 + radius + 30;
    } else {
        centerX = radius + 30;
    }
    int centerY = radius + 20;  // 底部留一點空間

    // 1. 繪製黑色外圈
    glColor4f(0.0f, 0.0f, 0.0f, 0.8f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(centerX, centerY);  // 中心點
    for (int i = 0; i <= 360; i += 5) {
        float angle = i * M_PI / 180.0f;
        glVertex2f(centerX + (radius + 5) * cosf(angle), centerY + (radius + 5) * sinf(angle));
    }
    glEnd();

    // 2. 繪製內部圓盤背景 - 略暗一點的灰黑色
    glColor4f(0.1f, 0.1f, 0.1f, 0.8f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(centerX, centerY);  // 中心點
    for (int i = 0; i <= 360; i += 5) {
        float angle = i * M_PI / 180.0f;
        glVertex2f(centerX + radius * cosf(angle), centerY + radius * sinf(angle));
    }
    glEnd();

    // 3. 繪製漸層色弧 (綠-黃-紅) - 從135度開始繞到45度，共繞270度
    const int startAngle = 135;
    const int endAngle = -135;  // 等於 225，但為了方便計算漸變使用負數

    glBegin(GL_TRIANGLE_STRIP);
    for (int i = startAngle; i >= endAngle; i -= 5) {
        float angle = i * M_PI / 180.0f;

        // 計算顏色漸變 (綠->黃->紅)
        float t = (float)(startAngle - i) / (startAngle - endAngle);  // 0.0 到 1.0
        float r, g, b;
        if (t < 0.5f) {  // 綠到黃
            r = t * 2.0f;
            g = 1.0f;
            b = 0.0f;
        } else {  // 黃到紅
            r = 1.0f;
            g = 1.0f - (t - 0.5f) * 2.0f;
            b = 0.0f;
        }

        // 內外點，形成環
        glColor4f(r, g, b, 0.9f);
        glVertex2f(centerX + radius * 0.8f * cosf(angle), centerY + radius * 0.8f * sinf(angle));    // 內環
        glVertex2f(centerX + radius * 0.95f * cosf(angle), centerY + radius * 0.95f * sinf(angle));  // 外環
    }
    glEnd();

    // 4. 繪製當前速度的填充扇形
    float maxSpeed = MAX_F_ROAD * 3.6f;  // 最高速度，換算成km/h
    float speedRatio = min(1.0f, kmh / maxSpeed);
    float currentAngle = startAngle - speedRatio * (startAngle - endAngle);

    // 繪製半透明填充扇形指示當前速度
    glBegin(GL_TRIANGLE_FAN);
    // 顏色根據當前速度
    if (speedRatio < 0.5f) {  // 低-中速
        glColor4f(speedRatio * 2.0f, 1.0f, 0.0f, 0.5f);
    } else {  // 中-高速
        glColor4f(1.0f, 2.0f * (1.0f - speedRatio), 0.0f, 0.5f);
    }

    glVertex2f(centerX, centerY);  // 圓心

    // 從起始角到當前角度的扇形
    for (int i = startAngle; i >= currentAngle; i -= 5) {
        float angle = i * M_PI / 180.0f;
        glVertex2f(centerX + radius * 0.75f * cosf(angle), centerY + radius * 0.75f * sinf(angle));
    }
    glEnd();

    // 繪製數字和文字
    char buf[16];
    sprintf(buf, "%.0f", kmh);

    // 繪製速度數字
    glColor3f(1.0f, 1.0f, 1.0f);  // 白色文字更清晰
    // 居中顯示時速
    int textWidth = 0;
    // 使用較大的字體
    for (char* p = buf; *p; ++p) textWidth += glutBitmapWidth(GLUT_BITMAP_TIMES_ROMAN_24, *p);
    glRasterPos2i(centerX - textWidth / 2, centerY - 8);  // 位置微調

    for (char* p = buf; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *p);

    // RACING 文字
    const char* racingText = "RACING";
    textWidth = 0;
    // 縮小RACING字體
    for (const char* p = racingText; *p; ++p) textWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_12, *p);
    glRasterPos2i(centerX - textWidth / 2, centerY - 30);  // 位置微調

    glColor3f(1.0f, 1.0f, 1.0f);  // 白色
    for (const char* p = racingText; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p);

    // km/h 標籤
    const char* unitText = "km/h";
    textWidth = 0;
    // 放大km/h字體
    for (const char* p = unitText; *p; ++p) textWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_18, *p);
    glRasterPos2i(centerX - textWidth / 2, centerY + 20);  // 位置微調

    glColor3f(1.0f, 1.0f, 1.0f);  // 白色
    for (const char* p = unitText; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// 集中初始化程式碼
void resetRace(bool startTimer = true) {
    initializeTrafficCones();
    timerRunning = startTimer;
    raceTime = 0.0f;
    firstFrame = true;

    travelP1 = travelP2 = 0.0f;
    lastCheckpointP1 = lastCheckpointP2 = 0;
    lapP1 = lapP2 = 0;

    // 重置完成狀態
    p1Finished = p2Finished = false;
    finalTimeP1 = finalTimeP2 = 0.0f;
    finalRankP1 = finalRankP2 = 0;

    car1WasInFinishVolume = false;
    car2WasInFinishVolume = false;
    // 重置強制結算計時器
    forceSettlementActive = false;
    forceSettlementTimer = 0.0f;
    car.log.clear();             // 清空舊的狀態日誌
    car.recordedInputs.clear();  // 清空之前記錄的按鍵操作 (如果是 Solo 模式，這會在 keyDown/Up 中填充)
    car.isAIControlledByReplay = false;
    car.isAIControlledByInputReplay = false;  // 新增
    car.replayLogToFollow.clear();
    car.currentReplayFrameIndex = 0;
    car.currentReplayInputActionIndex = 0;  // 新增

    car2.log.clear();
    // 計算起點方向和垂直方向向量
    Vec3 dir = normalize(trackSamples[1] - trackSamples[0]);
    Vec3 perpDir = Vec3(-dir.z, 0, dir.x);  // 計算垂直於行進方向的向量

    // Set car positions based on game mode
    if (gameMode == MODE_AI) {
        car.isAIControlledByInputReplay = true;  // AI 由按鍵記錄控制
        loadTopSoloReplays();                    // 載入 solo_replays.txt 獲取檔案列表
        if (!topSoloReplays.empty()) {
            // BUG FIX: Use loadInputReplayLogFromFile and car.recordedInputs for .inputrec files
            // Also, ensure the correct AI control flag is reset on failure.
            if (!loadInputReplayLogFromFile(topSoloReplays[0].logFilePath, car.recordedInputs)) {
                cerr << "[WARNING] Failed to load best solo input replay for AI control. Car (P1) will be idle."
                     << endl;
                car.isAIControlledByInputReplay = false;  // 載入失敗，取消 AI (input replay) 控制
            } else {
                cout << "[INFO] Car (P1) will be controlled by input replay: " << topSoloReplays[0].logFilePath << endl;
                // 重置車輛到起跑線狀態，因為 replay 是從比賽開始記錄的
                Vec3 dir_ai = normalize(trackSamples[1] - trackSamples[0]);  // Renamed to dir_ai to avoid conflict
                Vec3 perpDir_ai = Vec3(-dir_ai.z, 0, dir_ai.x);              // Renamed to perpDir_ai
                car.x = trackSamples[0].x - perpDir_ai.x * 4.0f;             // 與PVP模式下的P1起始位置一致
                car.y = trackSamples[0].y;
                car.z = trackSamples[0].z - perpDir_ai.z * 4.0f;
                car.heading = atan2f(dir_ai.x, dir_ai.z) * 180.0f / M_PI;
                car.speed = 0.0f;
                car.kW = car.kS = car.kA = car.kD = false;  // 確保初始按鍵狀態為false
                car.current_steer_force = 0.0f;
                // No need to clear car.recordedInputs here as it was just populated.
                // car.replayLogToFollow.clear(); // Not used for input replay
                // car.currentReplayFrameIndex = 0; // Not used for input replay
                car.currentReplayInputActionIndex = 0;  // Reset index for the new input log
            }
        } else {
            cerr << "[WARNING] No solo replays available for AI control. Car (P1) will be idle." << endl;
            car.isAIControlledByInputReplay = false;  // 沒有記錄，取消 AI (input replay) 控制
        }
        // For P2 (human player in 1vPC)
        Vec3 dir_p2_ai = normalize(trackSamples[1] - trackSamples[0]);  // Renamed for clarity
        Vec3 perpDir_p2_ai = Vec3(-dir_p2_ai.z, 0, dir_p2_ai.x);        // Renamed for clarity
        car2.x = trackSamples[0].x + perpDir_p2_ai.x * 4.0f;
        car2.y = trackSamples[0].y;
        car2.z = trackSamples[0].z + perpDir_p2_ai.z * 4.0f;
        car2.heading = atan2f(dir_p2_ai.x, dir_p2_ai.z) * 180.0f / M_PI;
        car2.speed = 0.0f;
        car2.kW = car2.kS = car2.kA = car2.kD = false;
        car2.current_steer_force = 0.0f;
        car2.log.clear();
        car2.recordedInputs.clear();  // P2 is human controlled, clear any previous inputs
        car2.isAIControlledByInputReplay = false;

    } else if (gameMode == MODE_SOLO) {                                // Solo 模式下，玩家控制，準備記錄按鍵
        Vec3 dir_solo = normalize(trackSamples[1] - trackSamples[0]);  // Renamed
        car.x = trackSamples[0].x;                                     // Solo 模式車輛在中間
        car.y = trackSamples[0].y;
        car.z = trackSamples[0].z;
        car.heading = atan2f(dir_solo.x, dir_solo.z) * 180.0f / M_PI;
        car.speed = 0.0f;
        car.kW = car.kS = car.kA = car.kD = false;
        car.current_steer_force = 0.0f;
        // car.recordedInputs.clear(); // Already cleared at the beginning of resetRace
        // car.isAIControlledByInputReplay = false; // Already cleared

        // Car2 is not used in Solo mode, but reset its state for consistency if needed elsewhere
        car2.x = trackSamples[0].x + 10000.0f;  // Move P2 far away or disable
        car2.y = trackSamples[0].y;
        car2.z = trackSamples[0].z;
        car2.speed = 0.0f;

    } else {                                                          // PVP 等其他模式 (P1 and P2 are human controlled)
        Vec3 dir_pvp = normalize(trackSamples[1] - trackSamples[0]);  // Renamed
        Vec3 perpDir_pvp = Vec3(-dir_pvp.z, 0, dir_pvp.x);            // Renamed

        // P1
        car.x = trackSamples[0].x - perpDir_pvp.x * 4.0f;
        car.y = trackSamples[0].y;
        car.z = trackSamples[0].z - perpDir_pvp.z * 4.0f;
        car.heading = atan2f(dir_pvp.x, dir_pvp.z) * 180.0f / M_PI;
        car.speed = 0.0f;
        car.kW = car.kS = car.kA = car.kD = false;
        car.current_steer_force = 0.0f;
        // car.recordedInputs.clear(); // Cleared at function start
        // car.isAIControlledByInputReplay = false; // Cleared at function start

        // P2
        car2.x = trackSamples[0].x + perpDir_pvp.x * 4.0f;
        car2.y = trackSamples[0].y;
        car2.z = trackSamples[0].z + perpDir_pvp.z * 4.0f;
        car2.heading = atan2f(dir_pvp.x, dir_pvp.z) * 180.0f / M_PI;
        car2.speed = 0.0f;
        car2.kW = car2.kS = car2.kA = car2.kD = false;
        car2.current_steer_force = 0.0f;
        // car2.recordedInputs.clear(); // Cleared at function start
        // car2.isAIControlledByInputReplay = false; // Cleared at function start
    }
    // car2 的設置
    Vec3 dir_c2 = normalize(trackSamples[1] - trackSamples[0]);
    Vec3 perpDir_c2 = Vec3(-dir_c2.z, 0, dir_c2.x);
    car2.x = trackSamples[0].x + perpDir_c2.x * 4.0f;
    car2.y = trackSamples[0].y;
    car2.z = trackSamples[0].z + perpDir_c2.z * 4.0f;
    car2.heading = atan2f(dir_c2.x, dir_c2.z) * 180.0f / M_PI;
    car2.speed = 0.0f;
    car2.kW = car2.kS = car2.kA = car2.kD = false;
    car2.current_steer_force = 0.0f;
    if (gameMode == MODE_SOLO) {
        // In solo mode, position the car at the center of the track
        car.x = trackSamples[0].x;
        car.y = trackSamples[0].y;
        car.z = trackSamples[0].z;
    } else {
        // In multiplayer modes, position cars side by side
        // 設置第一輛車在起跑線左側
        car.x = trackSamples[0].x - perpDir.x * 4.0f;
        car.y = trackSamples[0].y;
        car.z = trackSamples[0].z - perpDir.z * 4.0f;
    }

    // Set car heading and reset speed
    car.heading = atan2f(dir.x, dir.z) * 180.0f / M_PI;
    car.speed = 0.0f;
    car.log.clear();  // 清空紀錄

    // 設置第二輛車在起跑線右側
    car2.x = trackSamples[0].x + perpDir.x * 4.0f;
    car2.y = trackSamples[0].y;
    car2.z = trackSamples[0].z + perpDir.z * 4.0f;
    car2.heading = atan2f(dir.x, dir.z) * 180.0f / M_PI;
    car2.speed = 0.0f;
    car2.log.clear();
}

void UpdateCamera(float dt) {
    if (isMouseDragging) return;

    float carRad = car.heading * M_PI / 180.0f;
    float speedRatio = min(1.0f, max(0.0f, car.speed / MAX_F_ROAD));
    float dynamicDist = MAX_CAM_DIST - speedRatio * (MAX_CAM_DIST - MIN_CAM_DIST);
    float dynamicHeight = CAM_H - speedRatio * 1.0f;

    Vec3 desired_pos = {car.x - sinf(carRad) * dynamicDist, car.y + dynamicHeight, car.z - cosf(carRad) * dynamicDist};
    Vec3 raw_look_target = {car.x + sinf(carRad) * LOOK_AHEAD, car.y + 1.0f, car.z + cosf(carRad) * LOOK_AHEAD};
    float target_fov = BASE_FOV + (MAX_FOV - BASE_FOV) * speedRatio;

    float posSmoothFactor = 1.0f - exp(-CAM_POS_SMOOTH_RATE * dt);
    cam.pos = cam.pos + (desired_pos - cam.pos) * posSmoothFactor;

    Vec3 car_anchor_point = {car.x, car.y + dynamicHeight * 0.5f, car.z};
    Vec3 vec_car_to_cam = cam.pos - car_anchor_point;
    float dist_car_to_cam = length(vec_car_to_cam);
    float max_allowed_leash_distance = dynamicDist + MAX_DIST_LEASH_TOLERANCE;

    if (dist_car_to_cam > max_allowed_leash_distance) {
        cam.pos = car_anchor_point + normalize(vec_car_to_cam) * max_allowed_leash_distance;
    }

    float lookAtSmoothFactor = 1.0f - exp(-CAM_LOOK_SMOOTH_RATE * dt);
    cam.lookTarget = cam.lookTarget + (raw_look_target - cam.lookTarget) * lookAtSmoothFactor;

    float fovSmoothFactor = 1.0f - exp(-CAM_FOV_SMOOTH_RATE * dt);
    cam.fov = cam.fov + (target_fov - cam.fov) * fovSmoothFactor;

    // 統一的攝影機Y座標下限限制
    if (cam.pos.y < 0.5f) cam.pos.y = 0.5f;
}

// 修改 UpdateCamera2() 函數
void UpdateCamera2(float dt) {
    if (isMouseDragging) return;

    float car2Rad = car2.heading * M_PI / 180.0f;
    float speedRatio2 = min(1.0f, max(0.0f, car2.speed / MAX_F_ROAD));
    float dynamicDist2 = MAX_CAM_DIST - speedRatio2 * (MAX_CAM_DIST - MIN_CAM_DIST);
    float dynamicHeight2 = CAM_H - speedRatio2 * 1.0f;

    Vec3 desired_pos2 = {car2.x - sinf(car2Rad) * dynamicDist2, car2.y + dynamicHeight2,
                         car2.z - cosf(car2Rad) * dynamicDist2};
    Vec3 raw_look_target2 = {car2.x + sinf(car2Rad) * LOOK_AHEAD, car2.y + 1.0f, car2.z + cosf(car2Rad) * LOOK_AHEAD};
    float target_fov2 = BASE_FOV + (MAX_FOV - BASE_FOV) * speedRatio2;

    float posSmoothFactor2 = 1.0f - exp(-CAM_POS_SMOOTH_RATE * dt);
    cam2.pos = cam2.pos + (desired_pos2 - cam2.pos) * posSmoothFactor2;

    Vec3 car2_anchor_point = {car2.x, car2.y + dynamicHeight2 * 0.5f, car2.z};
    Vec3 vec_car2_to_cam2 = cam2.pos - car2_anchor_point;
    float dist_car2_to_cam2 = length(vec_car2_to_cam2);
    float max_allowed_leash_distance2 = dynamicDist2 + MAX_DIST_LEASH_TOLERANCE;

    if (dist_car2_to_cam2 > max_allowed_leash_distance2) {
        cam2.pos = car2_anchor_point + normalize(vec_car2_to_cam2) * max_allowed_leash_distance2;
    }

    float lookAtSmoothFactor2 = 1.0f - exp(-CAM_LOOK_SMOOTH_RATE * dt);
    cam2.lookTarget = cam2.lookTarget + (raw_look_target2 - cam2.lookTarget) * lookAtSmoothFactor2;

    float fovSmoothFactor2 = 1.0f - exp(-CAM_FOV_SMOOTH_RATE * dt);
    cam2.fov = cam2.fov + (target_fov2 - cam2.fov) * fovSmoothFactor2;

    // 統一的攝影機Y座標下限限制
    if (cam2.pos.y < 0.5f) cam2.pos.y = 0.5f;
}

// ===== 新增：在畫面上顯示 WASD / 方向鍵 的控制提示 =====
void drawControlButtons(bool isLeftSide) {
    // 1) 取得目前按鍵狀態 --------------------------------------------
    bool pW, pA, pS, pD;
    if (gameMode == MODE_SOLO || isLeftSide) {
        pW = car.kW;
        pA = car.kA;
        pS = car.kS;
        pD = car.kD;
    } else {
        pW = car2.kW;
        pA = car2.kA;
        pS = car2.kS;
        pD = car2.kD;
    }

    // 2) 基本參數 --------------------------------------------------
    const int KEY_SZ = 40;       // 按鈕大小 (px)
    const int marginY = 30;      // 底部邊距
    const int halfW = winW / 2;  // 半屏寬

    // 中心 X (以各自半屏中央為基準) -------------------------------
    int centerX;
    if (gameMode == MODE_SOLO) {
        centerX = winW / 2;  // Solo mode: center of entire screen
    } else {
        centerX = isLeftSide ? (halfW / 2) : (halfW + halfW / 2);  // Split screen: center of each half
    }
    int baseY = marginY;  // 底列 Y 座標 (左下為 0)

    // Up/Down/Left/Right 按鈕左下角座標 ---------------------------
    int upX = centerX - KEY_SZ / 2;
    int upY = baseY + KEY_SZ;
    int leftX = upX - KEY_SZ;
    int leftY = baseY;
    int downX = upX;
    int downY = baseY;
    int rightX = upX + KEY_SZ;
    int rightY = baseY;

    auto drawKey = [&](int x, int y, const char* label, bool pressed) {
        // (a) 填滿 (按下時) --------------------------------------
        if (pressed) {
            glColor4f(0.25f, 0.8f, 0.25f, 0.8f);  // 綠色半透明
            glBegin(GL_QUADS);
            glVertex2i(x, y);
            glVertex2i(x + KEY_SZ, y);
            glVertex2i(x + KEY_SZ, y + KEY_SZ);
            glVertex2i(x, y + KEY_SZ);
            glEnd();
        }
        // (b) 白色邊框 ------------------------------------------
        glColor3f(1, 1, 1);
        glLineWidth(2);
        glBegin(GL_LINE_LOOP);
        glVertex2i(x, y);
        glVertex2i(x + KEY_SZ, y);
        glVertex2i(x + KEY_SZ, y + KEY_SZ);
        glVertex2i(x, y + KEY_SZ);
        glEnd();

        // (c) 文字 ----------------------------------------------
        int textW = 0;
        for (const char* p = label; *p; ++p) textW += glutBitmapWidth(GLUT_BITMAP_HELVETICA_18, *p);
        glRasterPos2i(x + (KEY_SZ - textW) / 2, y + KEY_SZ / 2 - 4);
        for (const char* p = label; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
    };

    // 3) 切換到 2D 投影 ------------------------------------------
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    // 4) 繪製四個按鈕 --------------------------------------------
    if (isLeftSide || gameMode == MODE_SOLO) {
        drawKey(upX, upY, "W", pW);
        drawKey(leftX, leftY, "A", pA);
        drawKey(downX, downY, "S", pS);
        drawKey(rightX, rightY, "D", pD);
    } else {
        drawKey(upX, upY, "^", pW);
        drawKey(leftX, leftY, "<", pA);
        drawKey(downX, downY, "v", pS);
        drawKey(rightX, rightY, ">", pD);
    }

    // 5) 還原狀態 -------------------------------------------------
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ====== 即時名次排名版 (左上角) ======
void drawLiveRanking(bool isLeftSide) {
    // Skip if we're not in a game mode that requires rankings
    if (!(gameMode == MODE_PVP || gameMode == MODE_AI)) return;  // Solo mode doesn't need rankings

    // 1) 判斷當前名次 -------------------------------------------
    struct Entry {
        int id;          // 1 或 2
        float progress;  // 進度值
        int lap;         // 圈數
        int checkpoint;  // 檢查點
    } e[2] = {{1, travelP1, lapP1, lastCheckpointP1}, {2, travelP2, lapP2, lastCheckpointP2}};

    // 若進度相等則維持原順序
    if (e[0].progress < e[1].progress) swap(e[0], e[1]);

    // 2) 切換至 2D 正交投影 -------------------------------------
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 3) 根據視窗大小動態調整排名板尺寸 --------------------------
    const float boxRatio = 0.22f;  // 排名板寬度佔視窗寬度的比例 (略增)
    const int effectiveWidth = (gameMode == MODE_PVP || gameMode == MODE_AI) ? winW / 2 : winW;
    const int boxW = max(220, (int)(effectiveWidth * boxRatio));  // 最小寬度增加
    const int rowH = max(30, (int)(winH * 0.04f));                // 行高略增
    const int boxH = rowH * 2 + 20;                               // 兩行加上上下間距

    const int boxX = isLeftSide ? 10 : winW / 2 + 10;
    const int boxY = winH - boxH - 10;
    // const int fontSize = max(16, (int)(winH * 0.025f)); // fontSize declared but not used directly for
    // glutBitmapCharacter

    // 4) 繪製漂亮的背景 -------------------------------------------
    // Shadow rendering removed as per request

    // 主體背景 - 使用更精緻的漸變
    glBegin(GL_QUADS);
    glColor4f(0.12f, 0.12f, 0.22f, 0.9f);  // 底部顏色 - 深藍色
    glVertex2i(boxX, boxY);
    glVertex2i(boxX + boxW, boxY);
    glColor4f(0.22f, 0.22f, 0.32f, 0.9f);  // 頂部顏色 - 稍亮
    glVertex2i(boxX + boxW, boxY + boxH);
    glVertex2i(boxX, boxY + boxH);
    glEnd();

    // 內部描邊 - 亮邊框效果
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    glColor4f(0.7f, 0.7f, 0.9f, 0.8f);  // 亮藍灰色
    glVertex2i(boxX + 2, boxY + 2);
    glVertex2i(boxX + boxW - 2, boxY + 2);
    glVertex2i(boxX + boxW - 2, boxY + boxH - 2);
    glVertex2i(boxX + 2, boxY + boxH - 2);
    glEnd();

    // 標題區背景 - 更明顯的對比
    const int titleH = rowH / 2 + 6;  // 稍微增加標題高度
    glBegin(GL_QUADS);
    glColor4f(0.25f, 0.25f, 0.4f, 0.95f);  // 左側顏色
    glVertex2i(boxX + 2, boxY + boxH - titleH);
    glColor4f(0.35f, 0.35f, 0.5f, 0.95f);  // 右側顏色 - 漸變效果
    glVertex2i(boxX + boxW - 2, boxY + boxH - titleH);
    glVertex2i(boxX + boxW - 2, boxY + boxH - 2);
    glColor4f(0.25f, 0.25f, 0.4f, 0.95f);  // 左側顏色
    glVertex2i(boxX + 2, boxY + boxH - 2);
    glEnd();

    // 繪製底部裝飾線 - 添加雙線效果
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor4f(0.5f, 0.5f, 0.8f, 0.7f);  // 藍紫色
    glVertex2i(boxX + 10, boxY + 6);
    glVertex2i(boxX + boxW - 10, boxY + 6);
    glEnd();

    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glColor4f(0.6f, 0.6f, 0.9f, 0.5f);  // 較淺藍紫色
    glVertex2i(boxX + 10, boxY + 9);
    glVertex2i(boxX + boxW - 10, boxY + 9);
    glEnd();

    // 5) 繪製標題文字 -------------------------------------------
    void* titleFont = GLUT_BITMAP_HELVETICA_18;
    const char* titleText = "RACE STANDINGS";
    int titleTextWidth = 0;
    for (const char* p = titleText; *p; ++p) {
        titleTextWidth += glutBitmapWidth(titleFont, *p);
    }
    int titleRenderX = boxX + (boxW - titleTextWidth) / 2;
    int titleRenderY = boxY + boxH - (titleH / 2) + 4;

    glColor3f(0.2f, 0.2f, 0.3f);
    glRasterPos2i(titleRenderX + 1, titleRenderY - 1);
    for (const char* p = titleText; *p; ++p) {
        glutBitmapCharacter(titleFont, *p);
    }
    glColor3f(1.0f, 1.0f, 0.7f);
    glRasterPos2i(titleRenderX, titleRenderY);
    for (const char* p = titleText; *p; ++p) {
        glutBitmapCharacter(titleFont, *p);
    }

    // 6) 繪製排名數據 - 改進布局 ------------------------------------
    const int colWidths[] = {(int)(boxW * 0.10f), (int)(boxW * 0.18f), (int)(boxW * 0.22f), (int)(boxW * 0.50f)};
    const int textOffsetY = rowH / 2 + 5;

    for (int i = 0; i < 2; ++i) {
        int currentRowY = boxY + boxH - titleH - (i + 1) * rowH - 5;
        int currentColX = boxX + 18;

        glBegin(GL_QUADS);
        if (i == 0) {
            glColor4f(0.28f, 0.28f, 0.45f, 0.8f);
            glVertex2i(boxX + 4, currentRowY);
            glVertex2i(boxX + boxW - 4, currentRowY);
            glColor4f(0.35f, 0.35f, 0.52f, 0.8f);
            glVertex2i(boxX + boxW - 4, currentRowY + rowH);
            glVertex2i(boxX + 4, currentRowY + rowH);
        } else {
            glColor4f(0.20f, 0.20f, 0.32f, 0.7f);
            glVertex2i(boxX + 4, currentRowY);
            glVertex2i(boxX + boxW - 4, currentRowY);
            glColor4f(0.26f, 0.26f, 0.38f, 0.7f);
            glVertex2i(boxX + boxW - 4, currentRowY + rowH);
            glVertex2i(boxX + 4, currentRowY + rowH);
        }
        glEnd();

        if (i == 0) {
            glBegin(GL_TRIANGLES);
            if (e[i].id == 1) {
                glColor4f(0.95f, 0.8f, 0.2f, 0.9f);
            } else {
                glColor4f(0.95f, 0.8f, 0.2f, 0.9f);
            }
            glVertex2i(currentColX - 8, currentRowY + rowH / 2 + 5);  // Adjusted X for icon based on currentColX
            glVertex2i(currentColX - 11, currentRowY + rowH / 2 - 2);
            glVertex2i(currentColX - 5, currentRowY + rowH / 2 - 2);
            glVertex2i(currentColX - 13, currentRowY + rowH / 2 + 3);
            glVertex2i(currentColX - 16, currentRowY + rowH / 2 - 2);
            glVertex2i(currentColX - 10, currentRowY + rowH / 2 - 2);
            glVertex2i(currentColX - 3, currentRowY + rowH / 2 + 3);
            glVertex2i(currentColX - 6, currentRowY + rowH / 2 - 2);
            glVertex2i(currentColX, currentRowY + rowH / 2 - 2);
            glEnd();
        }

        if (e[i].id == 1) {
            if (i == 0) {
                glColor3f(0.6f, 0.6f, 1.0f);
            } else {
                glColor3f(0.4f, 0.4f, 0.9f);
            }
        } else {
            if (i == 0) {
                glColor3f(1.0f, 0.6f, 0.6f);
            } else {
                glColor3f(0.9f, 0.4f, 0.4f);
            }
        }

        char buf[64];
        sprintf(buf, "%d", i + 1);
        glColor3f(0.1f, 0.1f, 0.1f);
        glRasterPos2i(currentColX + 1, currentRowY + textOffsetY - 1);
        for (char* p = buf; *p; ++p) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
        }
        if (i == 0) {
            glColor3f(1.0f, 0.9f, 0.3f);
        } else {
            glColor3f(0.8f, 0.8f, 0.8f);
        }
        glRasterPos2i(currentColX, currentRowY + textOffsetY);
        for (char* p = buf; *p; ++p) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
        }
        currentColX += colWidths[0];

        sprintf(buf, "P%d", e[i].id);
        glColor3f(0.1f, 0.1f, 0.1f);
        glRasterPos2i(currentColX + 1, currentRowY + textOffsetY - 1);
        for (char* p = buf; *p; ++p) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
        }
        if (e[i].id == 1) {
            if (i == 0)
                glColor3f(0.6f, 0.6f, 1.0f);
            else
                glColor3f(0.4f, 0.4f, 0.9f);
        } else {
            if (i == 0)
                glColor3f(1.0f, 0.6f, 0.6f);
            else
                glColor3f(0.9f, 0.4f, 0.4f);
        }
        glRasterPos2i(currentColX, currentRowY + textOffsetY);
        for (char* p = buf; *p; ++p) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
        }
        currentColX += colWidths[1];

        sprintf(buf, "LAP:%d", e[i].lap);
        glColor3f(0.1f, 0.1f, 0.1f);
        glRasterPos2i(currentColX + 1, currentRowY + textOffsetY - 1);
        for (char* p = buf; *p; ++p) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
        }
        glColor3f(0.95f, 0.95f, 0.95f);
        glRasterPos2i(currentColX, currentRowY + textOffsetY);
        for (char* p = buf; *p; ++p) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
        }
        currentColX += colWidths[2];

        float progressPercent = (trackSamples.empty() ? 0.0f : (e[i].checkpoint * 100.0f) / trackSamples.size());
        if (progressPercent > 100.0f) progressPercent = 100.0f;
        if (progressPercent < 0.0f) progressPercent = 0.0f;

        const int barHeight = 10;
        const int barBaseX = currentColX;
        const int barBaseY = currentRowY + (rowH - barHeight) / 2;

        int percentTextMaxWidthEstimate = 50;
        int gapBetweenBarAndText = 5;
        int paddingAtEndOfColumn = 5;

        int progressBarDrawableWidth =
            colWidths[3] - percentTextMaxWidthEstimate - gapBetweenBarAndText - paddingAtEndOfColumn;
        if (progressBarDrawableWidth < 10) {
            progressBarDrawableWidth = 10;
        }

        int textRenderX = barBaseX + progressBarDrawableWidth + gapBetweenBarAndText;

        glLineWidth(1.0f);
        glColor4f(0.1f, 0.1f, 0.1f, 0.7f);
        glBegin(GL_LINE_LOOP);
        glVertex2i(barBaseX, barBaseY);
        glVertex2i(barBaseX + progressBarDrawableWidth, barBaseY);
        glVertex2i(barBaseX + progressBarDrawableWidth, barBaseY + barHeight);
        glVertex2i(barBaseX, barBaseY + barHeight);
        glEnd();

        glColor4f(0.15f, 0.15f, 0.18f, 0.6f);
        glBegin(GL_QUADS);
        glVertex2i(barBaseX, barBaseY);
        glVertex2i(barBaseX + progressBarDrawableWidth, barBaseY);
        glVertex2i(barBaseX + progressBarDrawableWidth, barBaseY + barHeight);
        glVertex2i(barBaseX, barBaseY + barHeight);
        glEnd();

        int fillWidth = max(1, (int)(progressBarDrawableWidth * progressPercent / 100.0f));
        if (e[i].id == 1) {
            glBegin(GL_QUADS);
            glColor4f(0.2f, 0.2f, 0.9f, 0.9f);
            glVertex2i(barBaseX, barBaseY);
            glVertex2i(barBaseX, barBaseY + barHeight);
            glColor4f(0.5f, 0.5f, 1.0f, 0.9f);
            glVertex2i(barBaseX + fillWidth, barBaseY + barHeight);
            glVertex2i(barBaseX + fillWidth, barBaseY);
            glEnd();
            glBegin(GL_QUADS);
            glColor4f(0.7f, 0.7f, 1.0f, 0.4f);
            glVertex2i(barBaseX, barBaseY + barHeight - 2);
            glVertex2i(barBaseX + fillWidth, barBaseY + barHeight - 2);
            glColor4f(0.7f, 0.7f, 1.0f, 0.1f);
            glVertex2i(barBaseX + fillWidth, barBaseY + barHeight / 2);
            glVertex2i(barBaseX, barBaseY + barHeight / 2);
            glEnd();
        } else {
            glBegin(GL_QUADS);
            glColor4f(0.9f, 0.2f, 0.2f, 0.9f);
            glVertex2i(barBaseX, barBaseY);
            glVertex2i(barBaseX, barBaseY + barHeight);
            glColor4f(1.0f, 0.5f, 0.5f, 0.9f);
            glVertex2i(barBaseX + fillWidth, barBaseY + barHeight);
            glVertex2i(barBaseX + fillWidth, barBaseY);
            glEnd();
            glBegin(GL_QUADS);
            glColor4f(1.0f, 0.7f, 0.7f, 0.4f);
            glVertex2i(barBaseX, barBaseY + barHeight - 2);
            glVertex2i(barBaseX + fillWidth, barBaseY + barHeight - 2);
            glColor4f(1.0f, 0.7f, 0.7f, 0.1f);
            glVertex2i(barBaseX + fillWidth, barBaseY + barHeight / 2);
            glVertex2i(barBaseX, barBaseY + barHeight / 2);
            glEnd();
        }

        sprintf(buf, "%.1f%%", progressPercent);
        int currentPercentTextWidth = 0;
        for (char* p = buf; *p; ++p) {
            currentPercentTextWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_12, *p);
        }

        if (textRenderX + currentPercentTextWidth > barBaseX + colWidths[3] - paddingAtEndOfColumn) {
            textRenderX = barBaseX + colWidths[3] - paddingAtEndOfColumn - currentPercentTextWidth;
        }
        if (textRenderX < barBaseX + progressBarDrawableWidth + gapBetweenBarAndText) {
            textRenderX = barBaseX + progressBarDrawableWidth + gapBetweenBarAndText;
        }

        glColor3f(0.1f, 0.1f, 0.1f);
        glRasterPos2i(textRenderX + 1, currentRowY + textOffsetY - 1);
        for (char* p = buf; *p; ++p) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p);
        }
        glColor3f(1.0f, 1.0f, 1.0f);
        glRasterPos2i(textRenderX, currentRowY + textOffsetY);
        for (char* p = buf; *p; ++p) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p);
        }
    }

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

//---------------------------------------------------------------------------------
// 完整重置遊戲到主選單狀態
//---------------------------------------------------------------------------------
void fullResetToMenu() {
    setImeMode(true);
    if (musicPlaying) {
        stopBackgroundMusic();
    }
    if (isWindPlaying) {
        stopLoopingSound("wind");
        isWindPlaying = false;
    }
    initializeTrafficCones();
    // 1. 重置比賽相關狀態 (車輛位置, 計時器, 分數等)
    resetRace(false);  // false 表示不立即開始計時

    // 2. 重置遊戲模式
    gameMode = MODE_MENU;
    previousGameMode = MODE_MENU;  // 清除先前的遊戲模式紀錄
    targetGameMode = MODE_MENU;    // 清除載入目標
    countdownActive = false;       // 關閉倒數計時

    // 3. 重置玩家完成狀態和最終結果
    p1Finished = false;
    p2Finished = false;
    finalTimeP1 = 0.0f;
    finalTimeP2 = 0.0f;
    finalRankP1 = 0;
    finalRankP2 = 0;
    car1WasInFinishVolume = false;
    car2WasInFinishVolume = false;
    forceSettlementActive = false;
    forceSettlementTimer = 0.0f;
    //    確保 trackSamples 已經被 buildTrack() 初始化
    if (!trackSamples.empty()) {
        Vec3 initialCarPos = trackSamples[0];  // 假設 P1 的初始位置
        Vec3 initialDir = normalize(trackSamples[1] - trackSamples[0]);
        float initialHeadingRad = atan2f(initialDir.x, initialDir.z);

        // 重置 P1 攝影機
        cam.pos = {initialCarPos.x - sinf(initialHeadingRad) * CAM_DIST, initialCarPos.y + CAM_H,
                   initialCarPos.z - cosf(initialHeadingRad) * CAM_DIST};
        cam.lookTarget = {initialCarPos.x + sinf(initialHeadingRad) * LOOK_AHEAD, initialCarPos.y + 1.0f,
                          initialCarPos.z + cosf(initialHeadingRad) * LOOK_AHEAD};
        cam.fov = BASE_FOV;

        // 重置 P2 攝影機 (如果適用)
        Vec3 initialCar2PosOffset = {4.0f, 0.0f, 0.0f};     // 簡化P2的偏移量
        if (gameMode == MODE_PVP || gameMode == MODE_AI) {  // 只有多人模式才需要重置P2攝影機
            Vec3 perpDir = Vec3(-initialDir.z, 0, initialDir.x);
            Vec3 car2StartPos = trackSamples[0] + perpDir * 4.0f;

            cam2.pos = {car2StartPos.x - sinf(initialHeadingRad) * CAM_DIST, car2StartPos.y + CAM_H,
                        car2StartPos.z - cosf(initialHeadingRad) * CAM_DIST};
            cam2.lookTarget = {car2StartPos.x + sinf(initialHeadingRad) * LOOK_AHEAD, car2StartPos.y + 1.0f,
                               car2StartPos.z + cosf(initialHeadingRad) * LOOK_AHEAD};
            cam2.fov = BASE_FOV;
        }
    }

    // 5. 關閉設定選單
    isSettingsMenuOpen = false;
    // 更新音樂按鈕文字狀態 (因為 backgroundMusicEnabled 可能沒變，但按鈕文字需要刷新)
    btnToggleMusic.text = backgroundMusicEnabled ? "BGM: ON" : "BGM: OFF";
    // 重新載入排行榜分數 (如果需要的話，或者在進入 SOLO 模式時做)
    loadScores();

    // 重置滑鼠拖曳狀態
    isMouseDragging = false;
    startCameraRestoration = false;

    // 清除可能的預覽攝影機移動狀態
    previewCamMoveForward = false;
    previewCamMoveBackward = false;
    previewCamMoveLeft = false;
    previewCamMoveRight = false;

    // 強制重繪
    glutPostRedisplay();
}

//---------------------------------------------------------------------------------
// 繪製設定按鈕和彈出選單
//---------------------------------------------------------------------------------
void drawSettingsElements(int mouseX, int mouseY) {  // 傳入滑鼠座標以處理懸停
    // 設定2D正交投影
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // --- 1. 繪製設定圖示按鈕 ---
    btnSettingsIcon.x = winW - btnSettingsIcon.w - 15;  // X 座標設定在右側 (距離右邊界15px)
    btnSettingsIcon.y = 15;                             // Y 座標設定在底部 (距離底部15px)
    bool hoverSettingsIcon = (mouseX >= btnSettingsIcon.x && mouseX <= btnSettingsIcon.x + btnSettingsIcon.w &&
                              mouseY >= btnSettingsIcon.y && mouseY <= btnSettingsIcon.y + btnSettingsIcon.h);

    if (hoverSettingsIcon && !isSettingsMenuOpen) {
        glColor4f(0.4f, 0.6f, 0.9f, 0.9f);
    } else {
        glColor4f(0.25f, 0.25f, 0.3f, 0.8f);
    }
    glBegin(GL_QUADS);
    glVertex2i(btnSettingsIcon.x, btnSettingsIcon.y);
    glVertex2i(btnSettingsIcon.x + btnSettingsIcon.w, btnSettingsIcon.y);
    glVertex2i(btnSettingsIcon.x + btnSettingsIcon.w, btnSettingsIcon.y + btnSettingsIcon.h);
    glVertex2i(btnSettingsIcon.x, btnSettingsIcon.y + btnSettingsIcon.h);
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(1.5f);
    glBegin(GL_LINE_LOOP);
    glVertex2i(btnSettingsIcon.x, btnSettingsIcon.y);
    glVertex2i(btnSettingsIcon.x + btnSettingsIcon.w, btnSettingsIcon.y);
    glVertex2i(btnSettingsIcon.x + btnSettingsIcon.w, btnSettingsIcon.y + btnSettingsIcon.h);
    glVertex2i(btnSettingsIcon.x, btnSettingsIcon.y + btnSettingsIcon.h);
    glEnd();

    int settingsTextWidth = glutBitmapWidth(GLUT_BITMAP_HELVETICA_18, 'S');
    glRasterPos2i(btnSettingsIcon.x + (btnSettingsIcon.w - settingsTextWidth) / 2,
                  btnSettingsIcon.y + (btnSettingsIcon.h - 18) / 2 + 4);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'S');

    // --- 2. 如果設定選單開啟，繪製彈出框 ---
    if (isSettingsMenuOpen) {
        // (a) 半透明背景遮罩
        glColor4f(0.0f, 0.0f, 0.0f, 0.65f);
        glBegin(GL_QUADS);
        glVertex2i(0, 0);
        glVertex2i(winW, 0);
        glVertex2i(winW, winH);
        glVertex2i(0, winH);
        glEnd();

        // (b) 設定彈出框本身
        int boxW = 320;
        int boxH = 250;
        int boxX = (winW - boxW) / 2;
        int boxY = (winH - boxH) / 2;

        glColor4f(0.12f, 0.15f, 0.2f, 0.98f);
        glBegin(GL_QUADS);
        glVertex2i(boxX, boxY);
        glVertex2i(boxX + boxW, boxY);
        glVertex2i(boxX + boxW, boxY + boxH);
        glVertex2i(boxX, boxY + boxH);
        glEnd();

        glColor4f(0.5f, 0.5f, 0.7f, 1.0f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_LOOP);
        glVertex2i(boxX, boxY);
        glVertex2i(boxX + boxW, boxY);
        glVertex2i(boxX + boxW, boxY + boxH);
        glVertex2i(boxX, boxY + boxH);
        glEnd();

        // (c) 繪製 "Return to Menu & Reset" 按鈕 (位置調整)
        btnReturnToMenuFromSettings.x = boxX + (boxW - btnReturnToMenuFromSettings.w) / 2;
        btnReturnToMenuFromSettings.y = boxY + boxH - btnReturnToMenuFromSettings.h - 20;  // <---- 移到上方

        bool hoverReturnBtn = (mouseX >= btnReturnToMenuFromSettings.x &&
                               mouseX <= btnReturnToMenuFromSettings.x + btnReturnToMenuFromSettings.w &&
                               mouseY >= btnReturnToMenuFromSettings.y &&
                               mouseY <= btnReturnToMenuFromSettings.y + btnReturnToMenuFromSettings.h);
        drawUIButton(btnReturnToMenuFromSettings, hoverReturnBtn, 1.0f);

        // (d) 繪製 "Music: ON/OFF" 按鈕 (新按鈕) <---- 新增 ----
        btnToggleMusic.x = boxX + (boxW - btnToggleMusic.w) / 2;
        btnToggleMusic.y = btnReturnToMenuFromSettings.y - btnToggleMusic.h - 15;  // 在 Return 按鈕下方，間隔15px
        // 更新按鈕文字以反映當前狀態
        btnToggleMusic.text = backgroundMusicEnabled ? "Music: ON" : "Music: OFF";

        bool hoverToggleMusicBtn = (mouseX >= btnToggleMusic.x && mouseX <= btnToggleMusic.x + btnToggleMusic.w &&
                                    mouseY >= btnToggleMusic.y && mouseY <= btnToggleMusic.y + btnToggleMusic.h);
        drawUIButton(btnToggleMusic, hoverToggleMusicBtn, 1.0f);
        // (e) *** 新增：繪製 "SFX: ON/OFF" 按鈕 ***
        btnToggleSFX.x = boxX + (boxW - btnToggleSFX.w) / 2;
        btnToggleSFX.y = btnToggleMusic.y - btnToggleSFX.h - 15;  // 放在 BGM 按鈕下方
        btnToggleSFX.text = sfxEnabled ? "SFX: ON" : "SFX: OFF";

        bool hoverToggleSfxBtn = (mouseX >= btnToggleSFX.x && mouseX <= btnToggleSFX.x + btnToggleSFX.w &&
                                  mouseY >= btnToggleSFX.y && mouseY <= btnToggleSFX.y + btnToggleSFX.h);
        drawUIButton(btnToggleSFX, hoverToggleSfxBtn, 1.0f);
    }

    // 還原OpenGL狀態
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}

// ------------------------------------------------------------
//  GLUT callbacks
// ------------------------------------------------------------
// 輔助函數：檢查點是否在AABB內 (如果使用OBB，這裡會是轉換和檢測)
bool isPointInFinishVolume(const Vec3& carPos) {
    // 使用簡化的世界座標 AABB 檢測
    return (carPos.x >= finishVolumeMin.x && carPos.x <= finishVolumeMax.x && carPos.y >= finishVolumeMin.y &&
            carPos.y <= finishVolumeMax.y &&  // 通常車輛Y座標固定，但這裡保留
            carPos.z >= finishVolumeMin.z && carPos.z <= finishVolumeMax.z);
}
// ==== 車輛碰撞處理函式實作 ====
void handleCarCollision() {
    // 只在PVP或AI模式下處理碰撞，因為Solo模式只有一台車
    if (gameMode != MODE_PVP) return;

    // 計算兩車之間的向量和距離 (只考慮XZ平面)
    Vec3 distVec = {car.x - car2.x, 0, car.z - car2.z};
    float distance = length(distVec);

    // 檢查是否發生碰撞 (距離小於兩車半徑和)
    if (distance > 0 && distance < (CAR_COLLISION_RADIUS * 2)) {
        // --- 1. 計算碰撞法線和重疊量 ---
        Vec3 collisionNormal = normalize(distVec);
        float overlap = (CAR_COLLISION_RADIUS * 2) - distance;

        // --- 2. 位置修正 (將兩車推開，避免重疊) ---
        // 每輛車沿法線方向移開重疊量的一半
        car.x += collisionNormal.x * overlap * 0.5f;
        car.z += collisionNormal.z * overlap * 0.5f;
        car2.x -= collisionNormal.x * overlap * 0.5f;
        car2.z -= collisionNormal.z * overlap * 0.5f;

        // --- 3. 速度響應 (動量交換) ---
        // 將速度和方向轉換為速度向量
        float car1_rad = car.heading * M_PI / 180.0f;
        Vec3 v1 = {sinf(car1_rad) * car.speed, 0, cosf(car1_rad) * car.speed};

        float car2_rad = car2.heading * M_PI / 180.0f;
        Vec3 v2 = {sinf(car2_rad) * car2.speed, 0, cosf(car2_rad) * car2.speed};

        // 計算沿碰撞法線的速度分量 (純量)
        float v1n_scalar = v1.x * collisionNormal.x + v1.z * collisionNormal.z;
        float v2n_scalar = v2.x * collisionNormal.x + v2.z * collisionNormal.z;

        // 計算碰撞後沿法線的新速度 (假設質量相等)
        float new_v1n_scalar = v1n_scalar + (v2n_scalar - v1n_scalar) * COLLISION_RESTITUTION;
        float new_v2n_scalar = v2n_scalar + (v1n_scalar - v2n_scalar) * COLLISION_RESTITUTION;

        // 計算速度變化量
        Vec3 delta_v1 = collisionNormal * (new_v1n_scalar - v1n_scalar);
        Vec3 delta_v2 = collisionNormal * (new_v2n_scalar - v2n_scalar);

        // 更新速度向量
        v1 = v1 + delta_v1;
        v2 = v2 + delta_v2;

        // 將新的速度向量轉換回 speed 和 heading
        car.speed = length(v1);
        if (car.speed > 0.01f) {  // 避免速度過小時方向錯亂
            car.heading = atan2f(v1.x, v1.z) * 180.0f / M_PI;
        }

        car2.speed = length(v2);
        if (car2.speed > 0.01f) {
            car2.heading = atan2f(v2.x, v2.z) * 180.0f / M_PI;
        }
    }
}

const char* mciGetDeviceAlias() {
    return "";  // 暫時返回空字串，因為上面的邏輯不直接依賴它。
}

void updateCB() {
    int ms = glutGet(GLUT_ELAPSED_TIME);
    float dt = (ms - prevMs) * 0.001f;
    prevMs = ms;

    // FPS Calculation
    frameCount++;
    int timeInterval = ms - previousTime;
    if (timeInterval > 1000) {  // Update FPS every second
        fps = frameCount / (timeInterval / 1000.0f);
        previousTime = ms;
        frameCount = 0;
    }

    // ---- Background Music Logic ----
    static GameMode prevMusicMode = MODE_MENU;  // Initialize with the starting game mode

    if (prevMusicMode != gameMode && musicPlaying) {  // If mode changed AND music was playing
        stopBackgroundMusic();                        // Stop music from the previous mode
    }
    prevMusicMode = gameMode;  // Update the tracker to the current mode

    if (backgroundMusicEnabled && !musicPlaying) {  // If music is enabled and not currently playing
        if (gameMode == MODE_MENU) {
            playBackgroundMusic("Menu.mp3");
        } else if (gameMode == MODE_SOLO || gameMode == MODE_PVP || gameMode == MODE_AI) {
            playBackgroundMusic("game.mp3");  // Ensure game.mp3 is in the executable's directory
        } else if (gameMode == MODE_RESULT) {
            playBackgroundMusic("Result.mp3");  // Ensure Result.mp3 is in the executable's directory
        }
        // Other modes like LOADING, TRACK_PREVIEW will not start music here by default
    } else if (!backgroundMusicEnabled && musicPlaying) {  // If music is disabled but was playing
        stopBackgroundMusic();
    }
    // ---- End Background Music Logic ----

    // Handle mouse dragging and camera restoration
    static bool wasDraggingLastFrame = false;
    if (wasDraggingLastFrame && !isMouseDragging) {
        // Camera will naturally restore via UpdateCamera if not in preview mode
    }
    wasDraggingLastFrame = isMouseDragging;

    // Clamp maximum delta time
    const float MAX_DT = 0.05f;
    if (dt > MAX_DT) {
        dt = MAX_DT;
    }

    // Networking for AI mode
    if (gameMode == MODE_AI) {
        checkForConnections();
        handleClientCommands();
    }

    // Handle loading screen
    if (gameMode == MODE_LOADING) {
        loadingTime += dt;
        loadingProgress = min(1.0f, loadingTime / LOADING_DURATION);
        if (loadingProgress >= 1.0f) {
            // Important: Stop any currently playing music (e.g. menu music) before switching to game mode
            if (musicPlaying) {
                stopBackgroundMusic();
            }
            gameMode = targetGameMode;
            // previousGameMode = targetGameMode; // previousGameMode is set when entering RESULT
            resetRace(false);  // Don't start timer immediately
            countdownActive = true;
            countdownLeft = 4.0f;
            playSoundEffect("go.wav");
            setImeMode(false);
            // Music for the new targetGameMode will be handled in the next updateCB
        }
        glutPostRedisplay();
        return;
    }

    // For modes that primarily show static UI and music is handled above
    if (gameMode == MODE_MENU) {  // MODE_RESULT needs its logic below for key presses
        glutPostRedisplay();
        return;
    }

    if (gameMode == MODE_TRACK_PREVIEW) {
        float previewCamMoveSpeed = 100.0f * dt;
        Vec3 movementDelta = {0, 0, 0};
        if (previewCamMoveForward || previewCamMoveBackward || previewCamMoveLeft || previewCamMoveRight) {
            Vec3 viewDirection = normalize(cam.lookTarget - cam.pos);
            Vec3 forwardOnXZ, rightOnXZ;
            forwardOnXZ = normalize({viewDirection.x, 0.0f, viewDirection.z});
            if (length(forwardOnXZ) < 0.001f) {
                Vec3 actualCamRight = normalize(cross(viewDirection, {0.0f, 1.0f, 0.0f}));
                if (length(actualCamRight) < 0.001f) actualCamRight = {1.0f, 0.0f, 0.0f};
                forwardOnXZ = normalize(cross({0.0f, 1.0f, 0.0f}, actualCamRight));
                rightOnXZ = normalize({actualCamRight.x, 0.0f, actualCamRight.z});
                if (length(rightOnXZ) < 0.001f) rightOnXZ = {1.0f, 0.0f, 0.0f};
            } else {
                rightOnXZ = normalize({forwardOnXZ.z, 0.0f, -forwardOnXZ.x});
            }
            if (previewCamMoveForward) movementDelta = movementDelta + forwardOnXZ * previewCamMoveSpeed;
            if (previewCamMoveBackward) movementDelta = movementDelta - forwardOnXZ * previewCamMoveSpeed;
            if (previewCamMoveLeft) movementDelta = movementDelta - rightOnXZ * previewCamMoveSpeed;
            if (previewCamMoveRight) movementDelta = movementDelta + rightOnXZ * previewCamMoveSpeed;
        }
        if (length(movementDelta) > 0.0001f) {
            cam.pos = cam.pos + movementDelta;
            cam.lookTarget = cam.lookTarget + movementDelta;
        }
        glutPostRedisplay();
        return;
    }

    // Countdown logic
    if (countdownActive) {
        countdownLeft -= dt;  // 更新這一幀的剩餘時間
        if (countdownLeft <= 0.0f) {
            countdownActive = false;
            timerRunning = true;  // Start race timer
        }
    }

    // Car and game physics updates (only if not in countdown or if timer is running)
    if (!countdownActive || timerRunning) {
        car.update(dt);
        if (gameMode == MODE_PVP || gameMode == MODE_AI) {
            car2.update(dt);
        }
        handleCarCollision();
        updateEngineSound();
        if (gameMode == MODE_SOLO || gameMode == MODE_PVP || gameMode == MODE_AI) {
            handleCarConeCollision(car, dt);
            if (gameMode == MODE_PVP || gameMode == MODE_AI) {
                handleCarConeCollision(car2, dt);
            }
        }
    }
    if (gameMode == MODE_SOLO || gameMode == MODE_PVP || gameMode == MODE_AI) {
        updateTrafficCones(dt);
    }

    // Race timer
    if (timerRunning) raceTime += dt;

    // Track progress
    if (timerRunning && !trackSamples.empty()) {
        int checkpointP1 = nearestSampleIndex({car.x, car.y, car.z});
        if (checkpointP1 < lastCheckpointP1 && lastCheckpointP1 > (int)trackSamples.size() * 0.8f &&
            checkpointP1 < (int)trackSamples.size() * 0.2f) {
            // Lap increment handled by finish volume logic
        }
        lastCheckpointP1 = checkpointP1;
        travelP1 = lapP1 * trackSamples.size() + checkpointP1;

        if (gameMode == MODE_PVP || gameMode == MODE_AI) {
            int checkpointP2 = nearestSampleIndex({car2.x, car2.y, car2.z});
            if (checkpointP2 < lastCheckpointP2 && lastCheckpointP2 > (int)trackSamples.size() * 0.8f &&
                checkpointP2 < (int)trackSamples.size() * 0.2f) {
                // Lap increment handled by finish volume logic
            }
            lastCheckpointP2 = checkpointP2;
            travelP2 = lapP2 * trackSamples.size() + checkpointP2;
        }
    }

    // Finish line logic
    if (timerRunning && !trackSamples.empty()) {
        Vec3 car1Pos = {car.x, car.y, car.z};
        bool car1CurrentlyInFinishVolume = isPointInFinishVolume(car1Pos);

        if (!p1Finished && !car1WasInFinishVolume && car1CurrentlyInFinishVolume) {
            bool sufficientProgressP1 = ((lapP1 * trackSamples.size() + lastCheckpointP1) >=
                                         (LAPS_TO_FINISH * trackSamples.size() - trackSamples.size() * 0.25f));
            if (sufficientProgressP1) {
                lapP1++;
                if (lapP1 >= LAPS_TO_FINISH) {
                    finalTimeP1 = raceTime;
                    p1Finished = true;
                    if (gameMode == MODE_SOLO) saveSoloOperationAndManageReplays(finalTimeP1);
                    saveScore(finalTimeP1);

                    if (gameMode == MODE_SOLO)
                        finalRankP1 = 1;
                    else if (!p2Finished) {
                        finalRankP1 = 1;
                        finalRankP2 = 2;
                        forceSettlementActive = true;
                        forceSettlementTimer = FORCE_SETTLEMENT_TIME;
                    } else
                        finalRankP1 = 2;

                    bool allPlayersOrSoloDone =
                        (gameMode == MODE_SOLO && p1Finished) ||
                        ((gameMode == MODE_PVP || gameMode == MODE_AI) && p1Finished && p2Finished);
                    if (allPlayersOrSoloDone && gameMode != MODE_RESULT) {
                        if (musicPlaying) stopBackgroundMusic();  // Stop game music before result screen
                        previousGameMode = gameMode;              // Store the mode we came from
                        gameMode = MODE_RESULT;
                        forceSettlementActive = false;
                    }
                }
            }
        }
        car1WasInFinishVolume = car1CurrentlyInFinishVolume;

        if (gameMode == MODE_PVP || gameMode == MODE_AI) {
            Vec3 car2Pos = {car2.x, car2.y, car2.z};
            bool car2CurrentlyInFinishVolume = isPointInFinishVolume(car2Pos);
            if (!p2Finished && !car2WasInFinishVolume && car2CurrentlyInFinishVolume) {
                bool sufficientProgressP2 = ((lapP2 * trackSamples.size() + lastCheckpointP2) >=
                                             (LAPS_TO_FINISH * trackSamples.size() - trackSamples.size() * 0.25f));
                if (sufficientProgressP2) {
                    lapP2++;
                    if (lapP2 >= LAPS_TO_FINISH) {
                        finalTimeP2 = raceTime;
                        p2Finished = true;
                        if (!p1Finished) {
                            finalRankP2 = 1;
                            finalRankP1 = 2;
                            forceSettlementActive = true;
                            forceSettlementTimer = FORCE_SETTLEMENT_TIME;
                        } else
                            finalRankP2 = 2;

                        if (p1Finished && p2Finished && gameMode != MODE_RESULT) {
                            if (musicPlaying) stopBackgroundMusic();  // Stop game music
                            previousGameMode = gameMode;
                            gameMode = MODE_RESULT;
                            forceSettlementActive = false;
                        }
                    }
                }
            }
            car2WasInFinishVolume = car2CurrentlyInFinishVolume;
        }
    }

    // Force settlement timer
    if (forceSettlementActive) {
        forceSettlementTimer -= dt;
        if (forceSettlementTimer <= 0.0f) {
            if (gameMode != MODE_RESULT) {
                if (musicPlaying) stopBackgroundMusic();  // Stop game music
                previousGameMode = gameMode;
                gameMode = MODE_RESULT;
                forceSettlementActive = false;
                // Final rank adjustments for unfinished players
                if (!p1Finished) {
                    finalTimeP1 = 0.0f;
                    if (p2Finished)
                        finalRankP1 = 2;
                    else
                        finalRankP1 = (travelP1 >= travelP2 ? 1 : 2);
                }
                if (!p2Finished && (gameMode == MODE_PVP || gameMode == MODE_AI)) {
                    finalTimeP2 = 0.0f;
                    if (p1Finished)
                        finalRankP2 = 2;
                    else
                        finalRankP2 = (travelP2 > travelP1 ? 1 : 2);
                }
                if (finalRankP1 == 0 && finalRankP2 == 0) {  // Both unfinished by timeout
                    if (travelP1 >= travelP2) {
                        finalRankP1 = 1;
                        finalRankP2 = 2;
                    } else {
                        finalRankP1 = 2;
                        finalRankP2 = 1;
                    }
                } else if (finalRankP1 == 0 && p2Finished) {
                    finalRankP1 = 2;
                } else if (finalRankP2 == 0 && p1Finished) {
                    finalRankP2 = 2;
                }
            }
        }
    }

    // Camera updates
    if (!isMouseDragging && gameMode != MODE_TRACK_PREVIEW &&
        (gameMode == MODE_SOLO || gameMode == MODE_PVP || gameMode == MODE_AI)) {
        UpdateCamera(dt);
        if (gameMode == MODE_PVP || gameMode == MODE_AI) {
            UpdateCamera2(dt);
        }
    }
    glutPostRedisplay();
}

void displayCB() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);  // 確保混合啟用
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    int currentMouseX = lastMouseX;         // 使用全域的 lastMouseX (在 motionCB 中更新)
    int currentMouseY = winH - lastMouseY;  // 使用全域的 lastMouseY 並轉換座標

    if (gameMode == MODE_MENU) {
        glViewport(0, 0, winW, winH);
        drawMainMenu(currentMouseX, currentMouseY);
    } else if (gameMode == MODE_RESULT) {
        glViewport(0, 0, winW, winH);
        drawResultScreen();
    } else if (gameMode == MODE_LOADING) {
        drawLoadingScreen();
    } else if (gameMode == MODE_TRACK_PREVIEW) {
        glViewport(0, 0, winW, winH);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(cam.fov, (double)winW / winH, 5.0f, 20000.0);  // 增加遠平面距離
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(cam.pos.x, cam.pos.y, cam.pos.z, cam.lookTarget.x, cam.lookTarget.y, cam.lookTarget.z, 0, 1, 0);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        drawBackground();
        glEnable(GL_DEPTH_TEST);

        drawGrid();
        drawTrack3D();
        drawCurbstones();
        drawTrackBarriers();
        drawEnvironmentModels();  // 繪製環境模型
        drawTrafficCones();
        drawGate(gateStartPos, gateStartHdg, true);     // true 表示這是起點拱門
        drawGate(gateFinishPos, gateFinishHdg, false);  // false 表示這是終點拱門
        drawLaneMarkings();
        drawTrackPreviewUI();  // 繪製操作提示
    } else if (gameMode == MODE_PVP || gameMode == MODE_AI) {
        // First half for car1 (left side)
        glViewport(0, 0, winW / 2, winH);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(cam.fov, ((double)winW / 2) / winH, 0.5, 20000.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(cam.pos.x, cam.pos.y, cam.pos.z, cam.lookTarget.x, cam.lookTarget.y, cam.lookTarget.z, 0, 1, 0);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);  // 確保天空在背景
        drawBackground();
        glEnable(GL_DEPTH_TEST);
        drawGrid();
        drawTrack3D();
        drawCurbstones();
        drawTrackBarriers();
        drawEnvironmentModels();
        drawTrafficCones();
        drawGate(gateStartPos, gateStartHdg, true);
        drawGate(gateFinishPos, gateFinishHdg, false);
        drawFinishVolume();  // <--- 左視圖的終點區域
        drawLaneMarkings();
        drawCar3D();

        // Draw HUD elements for car1
        drawMiniMap(true);  // 左側視窗，傳入true
        drawHUDTimer(true);
        drawCountdown(true);
        drawSpeedometer(car.speed, true);  // 添加時速表
        drawControlButtons(true);          // 添加控制提示
        // 在左側視窗添加即時排名板
        drawLiveRanking(true);
        // ---- 添加左側強制結算倒計時 ----
        drawForceSettlementTimer(true);
        drawVerticalDivider();

        // Second half for car2 (right side)
        glViewport(winW / 2, 0, winW / 2, winH);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(cam2.fov, ((double)winW / 2) / winH, 0.5, 20000.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(cam2.pos.x, cam2.pos.y, cam2.pos.z, cam2.lookTarget.x, cam2.lookTarget.y, cam2.lookTarget.z, 0, 1, 0);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);  // 確保天空在背景
        drawBackground();
        glEnable(GL_DEPTH_TEST);

        drawGrid();
        drawTrack3D();
        drawCurbstones();
        drawTrackBarriers();
        drawEnvironmentModels();
        drawGate(gateStartPos, gateStartHdg, true);
        drawGate(gateFinishPos, gateFinishHdg, false);
        drawFinishVolume();  // <--- 右視圖的終點區域
        drawLaneMarkings();
        drawCar3D();

        // Draw HUD elements for car2
        drawMiniMap(false);  // 右側視窗，傳入false
        drawHUDTimer(false);
        drawCountdown(false);
        drawSpeedometer(car2.speed, false);  // 添加時速表
        drawControlButtons(false);           // 添加控制提示

        // ---- 畫即時排名板 ----
        drawLiveRanking(false);

        // ---- 畫強制結算倒計時 ----
        drawForceSettlementTimer(false);
    } else {  // Solo mode or other single viewport modes
        glViewport(0, 0, winW, winH);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(cam.fov, (double)winW / winH, 0.5, 20000.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(cam.pos.x, cam.pos.y, cam.pos.z, cam.lookTarget.x, cam.lookTarget.y, cam.lookTarget.z, 0, 1, 0);

        // 添加藍天 - 繪製天空盒頂部（僅上半部）
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);  // 確保天空在背景
        drawBackground();
        glEnable(GL_DEPTH_TEST);
        drawGrid();
        drawTrack3D();
        drawCurbstones();
        drawTrackBarriers();
        drawEnvironmentModels();
        drawTrafficCones();
        drawGate(gateStartPos, gateStartHdg, true);
        drawGate(gateFinishPos, gateFinishHdg, false);
        drawFinishVolume();  // <--- 在這裡加入繪製終點空間的函數調用
        drawLaneMarkings();
        drawCar3D();

        // Draw UI elements for SOLO mode
        if (gameMode == MODE_SOLO) {
            drawMiniMap(false);
            drawHUDTimer(false);
            drawCountdown(false);
            drawSpeedometer(car.speed, false);
            drawControlButtons(false);
            drawForceSettlementTimer(false);
        } else {
            // Other single-viewport modes (fallback)
            drawMiniMap(false);
            drawHUDTimer(false);
            drawCountdown(false);
            drawSpeedometer(car.speed, false);
            drawForceSettlementTimer(false);
        }
    }
    if (gameMode != MODE_LOADING) {
        if (!isSettingsMenuOpen || gameMode == MODE_MENU || gameMode == MODE_RESULT) {
            drawFPS();
        }
    }
    if (gameMode == MODE_MENU || gameMode == MODE_SOLO || gameMode == MODE_PVP || gameMode == MODE_AI ||
        gameMode == MODE_RESULT) {
        drawSettingsElements(currentMouseX, currentMouseY);
    }
    glDisable(GL_BLEND);
    glutSwapBuffers();
}

// --------- reshapeCB  -----------
void reshapeCB(int w, int h) {
    if (h == 0) h = 1;

    /* 1) 先記錄舊尺寸，再求比例 */
    float sx = (float)w / winW;  // 寬度倍率
    float sy = (float)h / winH;  // 高度倍率

    /* 2) 雲整體等比例縮放、平移 */
    for (auto& c : clouds) {
        c.x *= sx;
        c.y *= sy;
        c.w *= sx;      // 寬度跟著水平比例
        c.h *= sy;      // 高度跟著垂直比例
        c.speed *= sx;  // 速度用水平像素/秒
    }
    for (auto& d : dust) {
        d.x *= sx;
        d.y *= sy;
        d.r *= (sx + sy) * 0.5f;
        d.vx *= sx;  // 水平速度按寬度比例
        d.vy *= sy;  // 垂直速度按高度比例
    }

    /* 3) 更新全域視窗寬高 */
    winW = w;
    winH = h;
    if (!gFullScreen) {  // 只在「視窗模式」下更新
        gWinWidth = w;
        gWinHeight = h;
    }

    glViewport(0, 0, w, h);
}

void mouseCB(int button, int state, int x, int y) {
    int invertedY = winH - y;

    if (button == GLUT_LEFT_BUTTON) {  // Only handle left click
        if (state == GLUT_DOWN) {
            // --- Priority 1: Settings Menu Click Handling ---
            if (isSettingsMenuOpen) {
                // Check "Return & Reset" button
                if (x >= btnReturnToMenuFromSettings.x &&
                    x <= btnReturnToMenuFromSettings.x + btnReturnToMenuFromSettings.w &&
                    invertedY >= btnReturnToMenuFromSettings.y &&
                    invertedY <= btnReturnToMenuFromSettings.y + btnReturnToMenuFromSettings.h) {
                    playSoundEffect("clickButtons.wav");
                    fullResetToMenu();
                } else if (x >= btnToggleMusic.x && x <= btnToggleMusic.x + btnToggleMusic.w &&
                           invertedY >= btnToggleMusic.y &&
                           invertedY <= btnToggleMusic.y + btnToggleMusic.h) {  // Check "Music: ON/OFF" button
                    playSoundEffect("clickButtons.wav");
                    toggleBackgroundMusic();  // This will play/stop music based on current mode and new toggle state
                } else if (x >= btnToggleSFX.x && x <= btnToggleSFX.x + btnToggleSFX.w && invertedY >= btnToggleSFX.y &&
                           invertedY <= btnToggleSFX.y + btnToggleSFX.h) {
                    playSoundEffect("clickButtons.wav");
                    toggleSfx();
                } else {
                    // Clicked inside settings panel but not on a button, or outside panel
                    int boxW_settings = 320;
                    int boxH_settings = 180;  // Keep consistent with drawSettingsElements
                    int boxX_settings = (winW - boxW_settings) / 2;
                    int boxY_settings = (winH - boxH_settings) / 2;
                    // If click is outside the settings pop-up box, close it
                    if (!(x >= boxX_settings && x <= boxX_settings + boxW_settings && invertedY >= boxY_settings &&
                          invertedY <= boxY_settings + boxH_settings)) {
                        isSettingsMenuOpen = false;
                    }
                }
                glutPostRedisplay();
                return;  // Settings menu interaction handled
            }

            // --- Priority 2: Settings Icon Click Handling ---
            if (x >= btnSettingsIcon.x && x <= btnSettingsIcon.x + btnSettingsIcon.w &&
                invertedY >= btnSettingsIcon.y && invertedY <= btnSettingsIcon.y + btnSettingsIcon.h) {
                // Allow settings icon in most game modes
                if (gameMode == MODE_MENU || gameMode == MODE_SOLO || gameMode == MODE_PVP || gameMode == MODE_AI ||
                    gameMode == MODE_RESULT) {
                    playSoundEffect("clickButtons.wav");
                    isSettingsMenuOpen = !isSettingsMenuOpen;
                    glutPostRedisplay();
                    return;  // Settings icon click handled
                }
            }

            // --- Priority 3: Main Menu Button Click Handling ---
            if (gameMode == MODE_MENU) {
                GameMode newTargetMode = MODE_MENU;  // Default to no change
                bool changingToLoading = false;

                if (x >= btnSolo.x && x <= btnSolo.x + btnSolo.w && invertedY >= btnSolo.y &&
                    invertedY <= btnSolo.y + btnSolo.h) {
                    playSoundEffect("clickButtons.wav");
                    newTargetMode = MODE_SOLO;
                    changingToLoading = true;
                } else if (x >= btnPVP.x && x <= btnPVP.x + btnPVP.w && invertedY >= btnPVP.y &&
                           invertedY <= btnPVP.y + btnPVP.h) {
                    playSoundEffect("clickButtons.wav");
                    newTargetMode = MODE_PVP;
                    changingToLoading = true;
                } else if (x >= btnAI.x && x <= btnAI.x + btnAI.w && invertedY >= btnAI.y &&
                           invertedY <= btnAI.y + btnAI.h) {
                    playSoundEffect("clickButtons.wav");
                    newTargetMode = MODE_AI;
                    changingToLoading = true;
                } else if (x >= btnPreview.x && x <= btnPreview.x + btnPreview.w && invertedY >= btnPreview.y &&
                           invertedY <= btnPreview.y + btnPreview.h) {
                    // For track preview, music will be stopped by updateCB's logic
                    playSoundEffect("clickButtons.wav");
                    gameMode = MODE_TRACK_PREVIEW;
                    // Setup preview camera
                    float trackCenterX = (mapMinX + mapMaxX) / 2.0f;
                    float trackCenterZ = (mapMinZ + mapMaxZ) / 2.0f;
                    cam.pos = {trackCenterX, 150.0f, trackCenterZ + 250.0f};
                    cam.lookTarget = {trackCenterX, 0.0f, trackCenterZ};
                    cam.fov = 60.0f;
                    isMouseDragging = false;
                }

                if (changingToLoading) {
                    gameMode = MODE_LOADING;
                    targetGameMode = newTargetMode;
                    loadingTime = 0.0f;
                    loadingProgress = 0.0f;
                }
                glutPostRedisplay();
                return;  // Main menu click handled
            }

            // --- Priority 4: In-Game or Track Preview Mouse Dragging ---
            if (gameMode == MODE_TRACK_PREVIEW ||
                (gameMode != MODE_LOADING && gameMode != MODE_RESULT && gameMode != MODE_MENU && !isSettingsMenuOpen)) {
                isMouseDragging = true;
                lastMouseX = x;
                lastMouseY = y;
                startCameraRestoration = false;

                if (gameMode != MODE_TRACK_PREVIEW) {  // Save original camera only for game modes
                    originalCamPos = cam.pos;
                    originalLookTarget = cam.lookTarget;
                    originalFov = cam.fov;
                    if (gameMode == MODE_PVP || gameMode == MODE_AI) {
                        originalCam2Pos = cam2.pos;
                        originalLookTarget2 = cam2.lookTarget;
                        originalFov2 = cam2.fov;
                    }
                }
            }
        } else if (state == GLUT_UP) {  // Mouse button released
            if (isMouseDragging) {
                isMouseDragging = false;
                // Camera restoration will be handled by UpdateCamera/UpdateCamera2 in updateCB
            }
        }
    }
    glutPostRedisplay();
}

// 修改 motionCB() 函式
void motionCB(int x, int y) {
    if (isMouseDragging) {
        int dx = x - lastMouseX;
        int dy = y - lastMouseY;

        if (gameMode == MODE_TRACK_PREVIEW) {
            float panSpeed = 0.5f;
            Vec3 forward = normalize(cam.lookTarget - cam.pos);
            Vec3 worldUp = {0, 1, 0};
            Vec3 right = normalize(cross(forward, worldUp));

            if (length(right) < 1e-5) {
                Vec3 arbitraryNonParallel = {(fabs(forward.y) < 0.9f) ? 1.0f : 0.0f,
                                             (fabs(forward.x) < 0.9f) ? 1.0f : 0.0f, 0.0f};
                if (fabs(forward.y) > 0.9f) arbitraryNonParallel = {0.0f, 0.0f, 1.0f};
                right = normalize(cross(arbitraryNonParallel, forward));
                if (length(right) < 1e-5) right = {1.0f, 0.0f, 0.0f};
            }
            Vec3 up = normalize(cross(right, forward));

            Vec3 moveHorizontal = right * (-dx * panSpeed);
            Vec3 moveVertical = up * (dy * panSpeed);

            Vec3 newCamPos = cam.pos + moveHorizontal + moveVertical;
            Vec3 newLookTarget = cam.lookTarget + moveHorizontal + moveVertical;

            // 限制攝影機高度
            if (newCamPos.y < 0.1f) {
                float diffY = 0.1f - newCamPos.y;
                newCamPos.y = 0.1f;
                newLookTarget.y += diffY;  // 同時調整 lookTarget 的高度，保持相對視角
            }
            cam.pos = newCamPos;
            cam.lookTarget = newLookTarget;

        } else if (gameMode != MODE_MENU && gameMode != MODE_LOADING && gameMode != MODE_RESULT) {
            float sensitivityYaw = 0.25f;
            float sensitivityPitch = 0.25f;
            Camera* pCam = &cam;
            Car* pCar = &car;  // 宣告 pCar
            if ((gameMode == MODE_PVP || gameMode == MODE_AI) && x >= winW / 2) {
                pCam = &cam2;
                pCar = &car2;  // 為 pCar 賦值
            }
            Vec3 carFocusPoint = {pCar->x, pCar->y + 1.0f, pCar->z};
            Vec3 currentRelativePos = pCam->pos - carFocusPoint;
            float distanceToCar = length(currentRelativePos);
            if (distanceToCar < 0.1f) distanceToCar = CAM_DIST;

            float yawAngleDeltaRad = -dx * sensitivityYaw * (M_PI / 180.0f);
            Vec3 yawedRelativePos;
            yawedRelativePos.x =
                currentRelativePos.x * cosf(yawAngleDeltaRad) - currentRelativePos.z * sinf(yawAngleDeltaRad);
            yawedRelativePos.z =
                currentRelativePos.x * sinf(yawAngleDeltaRad) + currentRelativePos.z * cosf(yawAngleDeltaRad);
            yawedRelativePos.y = currentRelativePos.y;

            Vec3 viewDirectionAfterYaw = normalize(Vec3{0, 0, 0} - yawedRelativePos);
            Vec3 worldUp = {0.0f, 1.0f, 0.0f};
            Vec3 camRight = normalize(cross(viewDirectionAfterYaw, worldUp));
            if (length(camRight) < 1e-5) {
                camRight = {1.0f, 0.0f, 0.0f};
                if (viewDirectionAfterYaw.y > 0)
                    camRight = {1.0f, 0.0f, 0.0f};
                else
                    camRight = {-1.0f, 0.0f, 0.0f};
            }
            float pitchAngleDeltaRad = -dy * sensitivityPitch * (M_PI / 180.0f);
            Vec3 k_axis = camRight;
            Vec3 v_vec = yawedRelativePos;
            float cos_theta = cosf(pitchAngleDeltaRad);
            float sin_theta = sinf(pitchAngleDeltaRad);
            float dot_kv = k_axis.x * v_vec.x + k_axis.y * v_vec.y + k_axis.z * v_vec.z;
            Vec3 pitchedRelativePos;
            pitchedRelativePos.x = v_vec.x * cos_theta + (k_axis.y * v_vec.z - k_axis.z * v_vec.y) * sin_theta +
                                   k_axis.x * dot_kv * (1 - cos_theta);
            pitchedRelativePos.y = v_vec.y * cos_theta + (k_axis.z * v_vec.x - k_axis.x * v_vec.z) * sin_theta +
                                   k_axis.y * dot_kv * (1 - cos_theta);
            pitchedRelativePos.z = v_vec.z * cos_theta + (k_axis.x * v_vec.y - k_axis.y * v_vec.x) * sin_theta +
                                   k_axis.z * dot_kv * (1 - cos_theta);

            float finalDist = length(pitchedRelativePos);
            if (finalDist < 0.1f) finalDist = distanceToCar;

            float currentPitch = asinf(pitchedRelativePos.y / finalDist);
            const float MAX_PITCH_RAD = 85.0f * (M_PI / 180.0f);
            const float MIN_PITCH_RAD_LOOKING_DOWN = -89.0f * (M_PI / 180.0f);  // 允許看得更低，但不能完全垂直向下

            // 檢查新的攝影機 Y 座標是否會低於 0.1
            Vec3 tempNewCamPos = carFocusPoint + pitchedRelativePos;
            if (tempNewCamPos.y < 0.1f) {
                // 如果直接應用 pitchedRelativePos 會導致攝影機過低，
                // 我們需要調整 pitchedRelativePos.y 使 newCamPos.y 恰好為 0.1f
                pitchedRelativePos.y = 0.1f - carFocusPoint.y;
                // 重新調整 pitchedRelativePos 的長度以保持一致性 (如果需要的話)
                // 但更簡單的做法是直接限制最終的 pCam->pos.y
            }

            // 限制俯仰角 (這段邏輯主要防止攝影機翻轉)
            if (currentPitch > MAX_PITCH_RAD) {
                currentPitch = MAX_PITCH_RAD;
            } else if (currentPitch < MIN_PITCH_RAD_LOOKING_DOWN) {  // 使用新的更低的下限
                currentPitch = MIN_PITCH_RAD_LOOKING_DOWN;
            }

            float newY = finalDist * sinf(currentPitch);
            float newXZprojLength = finalDist * cosf(currentPitch);
            Vec3 xz_part_normalized = normalize({pitchedRelativePos.x, 0, pitchedRelativePos.z});
            if (length(xz_part_normalized) < 1e-5) {
                xz_part_normalized = normalize({yawedRelativePos.x, 0, yawedRelativePos.z});
                if (length(xz_part_normalized) < 1e-5) xz_part_normalized = {1.0f, 0.0f, 0.0f};
            }
            pitchedRelativePos.x = xz_part_normalized.x * newXZprojLength;
            pitchedRelativePos.y = newY;
            pitchedRelativePos.z = xz_part_normalized.z * newXZprojLength;

            pCam->pos = carFocusPoint + pitchedRelativePos;

            // 在設定 pCam->pos 後，再次強制檢查 Y 座標
            if (pCam->pos.y < 0.5f) {
                pCam->pos.y = 0.5f;
            }
            pCam->lookTarget = carFocusPoint;  // 視線目標點保持在車輛焦點
        }
        lastMouseX = x;
        lastMouseY = y;
    } else {
        lastMouseX = x;
        lastMouseY = y;
    }
    glutPostRedisplay();
}

void specialKeyCB(int key, int, int) {
    if (key == GLUT_KEY_F11) {
        if (!gFullScreen) {  // ---- 進入全螢幕 ----
            gWinPosX = glutGet(GLUT_WINDOW_X);
            gWinPosY = glutGet(GLUT_WINDOW_Y);
            gWinWidth = glutGet(GLUT_WINDOW_WIDTH);
            gWinHeight = glutGet(GLUT_WINDOW_HEIGHT);

            glutFullScreen();
            gFullScreen = true;
        } else {  // ---- 離開全螢幕 ----
            glutReshapeWindow(gWinWidth, gWinHeight);
            glutPositionWindow(gWinPosX, gWinPosY);
            gFullScreen = false;
        }
    }

    // Handle controls for the second car
    if (gameMode != MODE_MENU && gameMode != MODE_LOADING && !countdownActive) {
        switch (key) {
            case GLUT_KEY_UP:
                car2.kW = true;
                break;
            case GLUT_KEY_DOWN:
                car2.kS = true;
                break;
            case GLUT_KEY_LEFT:
                car2.kA = true;
                break;
            case GLUT_KEY_RIGHT:
                car2.kD = true;
                break;
            case GLUT_KEY_HOME: {
                // Reset car2 to the nearest track center point
                int idx = nearestSampleIndex({car2.x, car2.y, car2.z});
                Vec3 p = trackSamples[idx];
                Vec3 dir = normalize(trackSamples[(idx + 1) % trackSamples.size()] - p);

                car2.x = p.x;
                car2.y = p.y;
                car2.z = p.z;
                car2.heading = atan2f(dir.x, dir.z) * 180.0f / M_PI;
                car2.speed = 0.0f;
                break;
            }
        }
    }
}

void keyDown(unsigned char k, int, int) {
    if (k == 27) {                             // ASCII 27 = Esc
        if (gameMode == MODE_TRACK_PREVIEW) {  // 如果在賽道預覽模式
            gameMode = MODE_MENU;              // 返回主選單
            /// 重置預覽移動旗標
            previewCamMoveForward = false;
            previewCamMoveBackward = false;
            previewCamMoveLeft = false;
            previewCamMoveRight = false;
            // 可以選擇是否重置攝影機到遊戲的預設狀態
            // cam.pos = {car.x, car.y + CAM_H, car.z - CAM_DIST}; // 範例：回到車後視角
            // cam.lookTarget = {car.x, car.y + 1.0f, car.z};
            // cam.fov = BASE_FOV;
            glutPostRedisplay();
            return;
        }
        if (gFullScreen) {
            glutReshapeWindow(gWinWidth, gWinHeight);
            glutPositionWindow(gWinPosX, gWinPosY);
            gFullScreen = false;
        } else {
            // exit(0);
        }
        return;
    }
    if (gameMode == MODE_TRACK_PREVIEW) {
        switch (toupper(k)) {  // 使用 toupper 處理大小寫
            case 'W':
                previewCamMoveForward = true;
                break;
            case 'S':
                previewCamMoveBackward = true;
                break;
            case 'A':
                previewCamMoveLeft = true;
                break;
            case 'D':
                previewCamMoveRight = true;
                break;
        }
        glutPostRedisplay();  // 按下按鍵後要求重繪以更新畫面
        return;               // 在預覽模式下，處理完WASD後返回，不執行後續的車輛控制
    }
    if (gameMode == MODE_SOLO && timerRunning && !countdownActive) {  // 只在 Solo 比賽進行中記錄
        // 檢查是否為 P1 的控制鍵
        unsigned char upperK = toupper(k);
        if (upperK == 'W' || upperK == 'S' || upperK == 'A' || upperK == 'D') {
            // 避免重複記錄連續按住的狀態，只記錄狀態改變的瞬間
            bool alreadyPressed = false;
            if (upperK == 'W' && car.kW) alreadyPressed = true;
            if (upperK == 'S' && car.kS) alreadyPressed = true;
            if (upperK == 'A' && car.kA) alreadyPressed = true;
            if (upperK == 'D' && car.kD) alreadyPressed = true;

            if (!alreadyPressed) {
                car.recordedInputs.push_back({raceTime, upperK, true});
            }
        }
    }
    if (gameMode == MODE_AI && (car.isAIControlledByReplay || car.isAIControlledByInputReplay)) {
        // 如果需要一些特殊鍵依然有效 (例如 'P' 存檔，'E' 重置AI車輛位置 - 雖然AI車輛應該自己跑)
        // 可以在這裡處理
        if (toupper(k) == 'P') {  // 假設 'P' 依然可以存檔 (AI車的log)
            if (car.isAIControlledByReplay)
                car.saveLog("ai_controlled_car_log.csv");
            else
                car.saveLog();  // 或者PVP/Solo時的P1車
        }
        return;  // 阻止 WASD 控制 AI 控制的車
    }
    if (gameMode == MODE_MENU || gameMode == MODE_LOADING || countdownActive || gameMode == MODE_TRACK_PREVIEW)
        return;  // 選單和載入畫面下直接忽略
    if (gameMode == MODE_RESULT) {
        if (k == ' ' || k == '\r') {                  // Space 或 Enter
            if (musicPlaying) stopBackgroundMusic();  // <--- 從結果畫面返回選單前，停止結果音樂

            gameMode = MODE_MENU;
            previousGameMode = MODE_MENU;  // Reset previous game mode
        }
        glutPostRedisplay();  // 請求重繪
        return;
    }
    switch (toupper(k)) {
        case 'W':
            if (!(gameMode == MODE_AI && car.isAIControlledByInputReplay)) car.kW = true;
            break;
        case 'S':
            if (!(gameMode == MODE_AI && car.isAIControlledByInputReplay)) car.kS = true;
            break;
        case 'A':
            if (!(gameMode == MODE_AI && car.isAIControlledByInputReplay)) car.kA = true;
            break;
        case 'D':
            if (!(gameMode == MODE_AI && car.isAIControlledByInputReplay)) car.kD = true;
            break;
        case 'E': {
            // 找最近的中心線點
            int idx = nearestSampleIndex({car.x, car.y, car.z});
            Vec3 p = trackSamples[idx];
            Vec3 dir = normalize(trackSamples[(idx + 1) % trackSamples.size()] - p);

            car.x = p.x;
            car.y = p.y;
            car.z = p.z;
            car.heading = atan2f(dir.x, dir.z) * 180.0f / M_PI;
            car.speed = 0.0f;  // 重新起跑，可改成保留部分速度
            break;
        }
        case 'p':
        case 'P':
            car.saveLog();
            break;
    }
}

void keyUp(unsigned char k, int, int) {
    if (gameMode == MODE_TRACK_PREVIEW) {
        switch (toupper(k)) {
            case 'W':
                previewCamMoveForward = false;
                break;
            case 'S':
                previewCamMoveBackward = false;
                break;
            case 'A':
                previewCamMoveLeft = false;
                break;
            case 'D':
                previewCamMoveRight = false;
                break;
        }
        glutPostRedisplay();
        return;  // 在預覽模式下，處理完WASD後返回
    }

    if (gameMode == MODE_SOLO && timerRunning && !countdownActive) {  // 只在 Solo 比賽進行中記錄
        unsigned char upperK = toupper(k);
        if (upperK == 'W' || upperK == 'S' || upperK == 'A' || upperK == 'D') {
            car.recordedInputs.push_back({raceTime, upperK, false});
        }
    }
    if (gameMode == MODE_AI && (car.isAIControlledByReplay || car.isAIControlledByInputReplay)) {
        return;
    }
    if (gameMode == MODE_MENU || gameMode == MODE_LOADING || countdownActive) return;

    switch (toupper(k)) {  // 使用 toupper 處理大小寫
        case 'W':
            if (!(gameMode == MODE_AI && car.isAIControlledByInputReplay)) car.kW = false;
            break;
        case 'S':
            if (!(gameMode == MODE_AI && car.isAIControlledByInputReplay)) car.kS = false;
            break;
        case 'A':
            if (!(gameMode == MODE_AI && car.isAIControlledByInputReplay)) car.kA = false;
            break;
        case 'D':
            if (!(gameMode == MODE_AI && car.isAIControlledByInputReplay)) car.kD = false;
            break;
    }
}

void specialKeyUp(int key, int, int) {
    if (gameMode == MODE_MENU || gameMode == MODE_LOADING || countdownActive) return;

    switch (key) {
        case GLUT_KEY_UP:
            car2.kW = false;
            break;
        case GLUT_KEY_DOWN:
            car2.kS = false;
            break;
        case GLUT_KEY_LEFT:
            car2.kA = false;
            break;
        case GLUT_KEY_RIGHT:
            car2.kD = false;
            break;
    }
}

void initGL() {
    makeShadowMatrix();
    glEnable(GL_BLEND);  // 影子需用到
    // 載入野馬模型
    cout << "Loading Mustang model..." << endl;
    if (!loadMultiMaterialModel(mustangModel, "mustang_GT/mustang_GT.obj")) {
        cerr << "Could not load the mustang model. Exiting." << endl;
        exit(1);
    }
    // 載入城市模型
    cout << "Loading CasteliaCity model..." << endl;
    if (!loadMultiMaterialModel(CasteliaCity, "CasteliaCity/CasteliaCity.obj")) {
        cerr << "Could not load the CasteliaCity model. Exiting." << endl;
        exit(1);
    }
    // 載入山模型
    cout << "Loading Cliff model..." << endl;
    if (!loadMultiMaterialModel(Cliff, "Cliff/mount.blend1.obj")) {
        cerr << "Could not load the CasteliaCity model. Exiting." << endl;
        exit(1);
    }
    // 載入島嶼模型
    cout << "Loading Tropical Islands model..." << endl;
    if (!loadMultiMaterialModel(TropicalIslands, "Tropical Islands/Tropical Islands.obj")) {
        cerr << "Could not load the CasteliaCity model. Exiting." << endl;
        exit(1);
    }
    // 載入三角錐模型
    cout << "Loading Cone1 model..." << endl;
    if (!loadMultiMaterialModel(Cone1, "Cone1/Cone1_obj.obj")) {
        cerr << "Could not load the CasteliaCity model. Exiting." << endl;
    }
    // // 載入三角錐模型
    // cout << "Loading Cone2 model..." << endl;
    // if (!loadMultiMaterialModel(Cone2, "Cone2/Cone2_obj.obj")) {
    //     cerr << "Could not load the CasteliaCity model. Exiting." << endl;
    //     exit(1);
    // }
    buildTrack();
    initializeTrafficCones();
    stbi_set_flip_vertically_on_load(1);  // 讓 stb 在載入時自動上下翻轉
    int w, h, n;
    unsigned char* data = stbi_load("Speedcar.png", &w, &h, &n, 4);
    if (!data) {
        fprintf(stderr, "[ERR] cannot load Speedcar.png\n");
        exit(1);
    }

    glGenTextures(1, &texMenu);
    glBindTexture(GL_TEXTURE_2D, texMenu);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_image_free(data);

    // 加載結算畫面背景
    data = stbi_load("SettlementScreen.png", &w, &h, &n, 4);
    if (!data) {
        fprintf(stderr, "[ERR] cannot load SettlementScreen.png\n");
        // 錯誤處理：使用空紋理
        unsigned char empty[4] = {0, 0, 0, 255};
        glGenTextures(1, &texSettlement);
        glBindTexture(GL_TEXTURE_2D, texSettlement);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, empty);
    } else {
        glGenTextures(1, &texSettlement);
        glBindTexture(GL_TEXTURE_2D, texSettlement);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        stbi_image_free(data);
    }
    // 加載柏油路紋理
    stbi_set_flip_vertically_on_load(1);  // 柏油路紋理通常不需要翻轉，先設為 false
    data = stbi_load("blackRand.png", &w, &h, &n, 0);
    if (!data) {
        fprintf(stderr, "[ERR] cannot load blackRand.png\n");
    } else {
        glGenTextures(1, &texAsphalt);
        glBindTexture(GL_TEXTURE_2D, texAsphalt);
        GLenum format = (n == 3) ? GL_RGB : GL_RGBA;

        // 設定紋理參數為重複，這樣紋理才能鋪滿整個賽道
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        // 設定紋理過濾方式
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // 建立紋理與 Mipmaps
        gluBuild2DMipmaps(GL_TEXTURE_2D, format, w, h, format, GL_UNSIGNED_BYTE, data);

        stbi_image_free(data);
    }
    stbi_set_flip_vertically_on_load(1);  // 恢復設定，以防影響其他圖片

    // 加載沙地紋理
    stbi_set_flip_vertically_on_load(0);          // 沙地紋理通常不需要垂直翻轉，如果貼上後顛倒再改為 1
    data = stbi_load("sand.png", &w, &h, &n, 4);  // 強制以 RGBA (4通道) 格式載入

    if (!data) {
        fprintf(stderr, "[ERR] 無法載入 sand.png。請檢查檔案路徑和格式。\n");
    } else {
        fprintf(stdout, "[INFO] 已載入 sand.png: %d x %d, 原始通道數: %d, 已載入為 4 通道。\n", w, h, n);
        if ((w > 0 && (w & (w - 1)) != 0) || (h > 0 && (h & (h - 1)) != 0)) {  // 檢查是否為2的次方
            fprintf(stderr,
                    "[WARN] sand.png 的尺寸 (%d x %d) 不是2的次方。這可能會導致 gluBuild2DMipmaps 出現問題。\n"
                    "       建議將圖片尺寸修改為例如 256x256 或 512x512。\n",
                    w, h);
        }

        glGenTextures(1, &texSand);
        glBindTexture(GL_TEXTURE_2D, texSand);

        GLenum loaded_pixel_data_format_sand = GL_RGBA;
        GLenum internal_texture_format_sand = GL_RGBA;

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        gluBuild2DMipmaps(GL_TEXTURE_2D, internal_texture_format_sand, w, h, loaded_pixel_data_format_sand,
                          GL_UNSIGNED_BYTE, data);

        stbi_image_free(data);
        fprintf(stdout, "[INFO] texSand ID: %u\n", texSand);
    }
    stbi_set_flip_vertically_on_load(1);  // 恢復設定，以防影響其他圖片載入

    // 加載草地紋理
    stbi_set_flip_vertically_on_load(0);  // 草地紋理通常不需要翻轉
    data = stbi_load("Grass.png", &w, &h, &n, 0);
    if (!data) {
        fprintf(stderr, "[ERR] cannot load Grass.png\n");
    } else {
        glGenTextures(1, &texGrass);
        glBindTexture(GL_TEXTURE_2D, texGrass);
        GLenum format = (n == 3) ? GL_RGB : GL_RGBA;
        // 設定紋理重複
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // 設定紋理過濾與 Mipmaps
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        gluBuild2DMipmaps(GL_TEXTURE_2D, format, w, h, format, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
        fprintf(stdout, "[INFO] texGrass ID: %u\n", texGrass);
    }
    stbi_set_flip_vertically_on_load(1);  // 恢復設定，以防影響其他圖片

    // ---- 讓車輛停在賽道起點，並對準前進方向 ----
    car.x = trackSamples[0].x;
    car.y = trackSamples[0].y;
    car.z = trackSamples[0].z;

    Vec3 dir = normalize(trackSamples[1] - trackSamples[0]);  // 起點→下一點
    car.heading = atan2f(dir.x, dir.z) * 180.0f / M_PI;       // 轉成度數
    car.speed = 0.0f;                                         // 靜止起跑
    /* ----- 產生雲 ----- */
    srand(2025);
    for (int i = 0; i < 5; ++i) {
        Cloud c;
        c.w = 120 + rand() % 80;  // 120~200px
        c.h = c.w * 0.6f;
        c.x = rand() % winW;
        c.y = winH * 0.75f + rand() % int(winH * 0.15f);
        c.speed = 15 + rand() % 25;  // 15~40 px/s
        clouds.push_back(c);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.52f, 0.8f, 0.92f, 1);
    loadScores();
    loadTopSoloReplays();  // 新增：載入 Solo 操作記錄排行榜
    dust.reserve(DUST_NUM);
    for (int i = 0; i < DUST_NUM; ++i) {
        Dust d;
        d.x = rand() % winW;
        d.y = rand() % int(winH * 0.45f);    // 下半部
        d.r = 2.0f + (rand() % 70) * 0.04f;  // 2‑5 px
        float s = 30.0f + rand() % 50;       // 30‑80 px/s 風速
        d.vx = s * 0.707f;                   // cos 45°
        d.vy = s * 0.707f;                   // sin 45°
        d.a = 0.15f + (rand() & 1) * 0.1f;
        dust.push_back(d);
    }
    cam.pos = {car.x, car.y + CAM_H, car.z - CAM_DIST};
    cam2.pos = {car2.x, car2.y + CAM_H, car2.z - CAM_DIST};

    // 初始化相機觀察目標點
    float carRad = car.heading * M_PI / 180.0f;
    cam.lookTarget.x = car.x + sinf(carRad) * LOOK_AHEAD;
    cam.lookTarget.y = car.y + 1.0f;
    cam.lookTarget.z = car.z + cosf(carRad) * LOOK_AHEAD;

    float car2Rad = car2.heading * M_PI / 180.0f;
    cam2.lookTarget.x = car2.x + sinf(car2Rad) * LOOK_AHEAD;
    cam2.lookTarget.y = car2.y + 1.0f;
    cam2.lookTarget.z = car2.z + cosf(car2Rad) * LOOK_AHEAD;
}

// Function to draw vertical divider between split screens
void drawVerticalDivider() {
    // Switch to 2D orthographic projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Disable depth test to draw over everything
    glDisable(GL_DEPTH_TEST);

    // Draw a white vertical line in the middle of the screen
    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glVertex2f(winW / 2.0f, 0);
    glVertex2f(winW / 2.0f, winH);
    glEnd();

    // Restore settings
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// Function to draw player labels
void drawPlayerLabel(int playerNum, bool isLeft) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);

    char label[20];
    sprintf(label, "Player %d", playerNum);

    // Calculate position - centered at top of each half
    int x = isLeft ? winW / 4 - 40 : 3 * winW / 4 - 40;
    int y = winH - 30;

    // Draw background box
    glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
    glBegin(GL_QUADS);
    glVertex2i(x - 10, y - 20);
    glVertex2i(x + 100, y - 20);
    glVertex2i(x + 100, y + 10);
    glVertex2i(x - 10, y + 10);
    glEnd();

    // Draw text
    glColor3f(playerNum == 1 ? 0.2f : 1.0f, 0.2f, playerNum == 2 ? 0.2f : 1.0f);  // Blue for P1, Red for P2
    glRasterPos2i(x, y);

    for (char* p = label; *p; ++p) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// 添加強制結算倒計時的顯示功能
void drawForceSettlementTimer(bool isLeftSide) {
    if (!forceSettlementActive) return;

    // 計算剩餘秒數
    int secLeft = (int)ceil(forceSettlementTimer);

    // 2D 正交投影設置
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    // 繪製半透明背景
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 根據剩餘時間設置顏色
    float alpha = 0.7f;
    if (secLeft <= 3) {
        // 最後3秒閃爍紅色
        alpha = 0.7f + 0.3f * sin(glutGet(GLUT_ELAPSED_TIME) * 0.01f);
        glColor4f(1.0f, 0.0f, 0.0f, alpha);  // 紅色警示
    } else {
        glColor4f(0.8f, 0.7f, 0.0f, alpha);  // 黃色提示
    }

    // 繪製通知框 - 根據分屏調整位置
    const int boxWidth = 220;  // 略微縮小以適應分屏
    const int boxHeight = 70;

    // 計算框位置 - 根據是左側還是右側視窗調整
    int boxX;
    if (gameMode == MODE_PVP || gameMode == MODE_AI) {
        // 分屏模式 - 將框放在各自視窗中央
        boxX = isLeftSide ? (winW / 4 - boxWidth / 2) : (winW * 3 / 4 - boxWidth / 2);
    } else {
        // 單一視窗模式 - 居中顯示
        boxX = (winW - boxWidth) / 2;
    }
    const int boxY = winH - 150;

    glBegin(GL_QUADS);
    glVertex2i(boxX, boxY);
    glVertex2i(boxX + boxWidth, boxY);
    glVertex2i(boxX + boxWidth, boxY + boxHeight);
    glVertex2i(boxX, boxY + boxHeight);
    glEnd();

    glDisable(GL_BLEND);

    // 繪製文字
    char buf[64];
    sprintf(buf, "RACE ENDING IN %d", secLeft);

    // 文字顏色
    glColor3f(1.0f, 1.0f, 1.0f);  // 白色文字

    // 計算文字長度以居中
    int textWidth = 0;
    for (char* p = buf; *p; ++p) {
        textWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_18, *p);
    }

    // 顯示倒計時文字
    glRasterPos2i(boxX + (boxWidth - textWidth) / 2, boxY + 40);
    for (char* p = buf; *p; ++p) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
    }

    // 還原狀態
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// Draw loading screen
void drawLoadingScreen() {
    glViewport(0, 0, winW, winH);

    // Set up 2D projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, winW, 0, winH);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Clear the screen with a dark background
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    // Draw "Loading..." text
    glColor3f(1.0f, 1.0f, 1.0f);
    const char* loadingText = "Loading...";

    // Calculate text position to be centered
    void* font = GLUT_STROKE_ROMAN;
    float textWidth = 0.0f;
    for (const char* p = loadingText; *p; ++p) {
        textWidth += glutStrokeWidth(font, *p);
    }
    const float textHeight = 119.05f;  // GLUT_STROKE_ROMAN height

    // Scale text to desired size
    const float textScale = winH * 0.1f / textHeight;
    const float scaledWidth = textWidth * textScale;

    // Position text in the center top of the screen
    const float textX = (winW - scaledWidth) * 0.5f;
    const float textY = winH * 0.7f;

    // Draw the text
    glPushMatrix();
    glTranslatef(textX, textY, 0);
    glScalef(textScale, textScale, 1.0f);
    for (const char* p = loadingText; *p; ++p) {
        glutStrokeCharacter(font, *p);
    }
    glPopMatrix();

    // Draw loading bar background
    const float barWidth = winW * 0.6f;
    const float barHeight = winH * 0.05f;
    const float barX = (winW - barWidth) * 0.5f;
    const float barY = winH * 0.5f - barHeight * 0.5f;

    glColor3f(0.3f, 0.3f, 0.3f);
    glBegin(GL_QUADS);
    glVertex2f(barX, barY);
    glVertex2f(barX + barWidth, barY);
    glVertex2f(barX + barWidth, barY + barHeight);
    glVertex2f(barX, barY + barHeight);
    glEnd();

    // Draw loading bar progress
    const float progressWidth = barWidth * loadingProgress;

    glColor3f(0.0f, 0.8f, 0.2f);
    glBegin(GL_QUADS);
    glVertex2f(barX, barY);
    glVertex2f(barX + progressWidth, barY);
    glVertex2f(barX + progressWidth, barY + barHeight);
    glVertex2f(barX, barY + barHeight);
    glEnd();

    // Draw loading tips based on target game mode
    const char* tipText = nullptr;
    if (targetGameMode == MODE_SOLO) {
        tipText = "Prepare for solo race! Use WASD to control your car.";
    } else if (targetGameMode == MODE_PVP) {
        tipText = "Get ready for PVP mode! Player 1: WASD, Player 2: Arrow Keys.";
    } else if (targetGameMode == MODE_AI) {
        tipText = "AI mode loading! Try to beat the computer opponent.";
    }

    if (tipText) {
        glColor3f(0.8f, 0.8f, 0.8f);

        // Calculate text position
        float tipWidth = 0.0f;
        for (const char* p = tipText; *p; ++p) {
            tipWidth += glutStrokeWidth(font, *p);
        }

        // Scale tip text to desired size
        const float tipScale = winH * 0.03f / textHeight;
        const float scaledTipWidth = tipWidth * tipScale;

        // Position tip text below the loading bar
        const float tipX = (winW - scaledTipWidth) * 0.5f;
        const float tipY = winH * 0.35f;

        // Draw the tip text
        glPushMatrix();
        glTranslatef(tipX, tipY, 0);
        glScalef(tipScale, tipScale, 1.0f);
        for (const char* p = tipText; *p; ++p) {
            glutStrokeCharacter(font, *p);
        }
        glPopMatrix();
    }

    // Restore GL state
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}
// AI模式通訊相關
SOCKET serverSocket = INVALID_SOCKET;
SOCKET clientSocket = INVALID_SOCKET;
bool clientConnected = false;

// 初始化網絡
bool initNetwork() {
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        cerr << "WSAStartup failed: " << result << "\n";
        return false;
    }
    serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSocket == INVALID_SOCKET) {
        cerr << "Error creating socket: " << WSAGetLastError() << "\n";
        WSACleanup();
        return false;
    }
    // 綁定套接字到本地地址和端口
    struct sockaddr_in service;
    service.sin_family = AF_INET;
    service.sin_addr.s_addr = inet_addr("127.0.0.1");
    service.sin_port = htons(12345);
    // 使用端口12345
    result = bind(serverSocket, (SOCKADDR*)&service, sizeof(service));
    if (result == SOCKET_ERROR) {
        cerr << "Bind failed: " << WSAGetLastError() << "\n";
        closesocket(serverSocket);
        WSACleanup();
        return false;
    }
    // 監聽連接請求
    result = listen(serverSocket, 1);
    if (result == SOCKET_ERROR) {
        cerr << "Listen failed: " << WSAGetLastError() << "\n";
        closesocket(serverSocket);
        WSACleanup();
        return false;
    }
    // 非阻塞模式
    u_long iMode = 1;
    ioctlsocket(serverSocket, FIONBIO, &iMode);
    cerr << "Server started, waiting for Python agent to connect...\n";
    return true;
}

// 檢查是否有新的連接請求

void checkForConnections() {
    if (clientConnected) return;
    struct sockaddr_in clientAddr;
    int clientAddrLen = sizeof(clientAddr);
    clientSocket = accept(serverSocket, (SOCKADDR*)&clientAddr, &clientAddrLen);
    if (clientSocket != INVALID_SOCKET) {
        cerr << "Python agent connected!\n";
        clientConnected = true;
        // 設置為非阻塞模式
        u_long iMode = 1;
        ioctlsocket(clientSocket, FIONBIO, &iMode);
    }
}

// 處理從Python代理接收到的命令

void handleClientCommands() {
    if (!clientConnected) return;
    char buffer[256] = {0};
    int bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    if (bytesReceived > 0) {
        buffer[bytesReceived] = '\0';
        // 確保字符串終止
        // 處理GET_STATE命令 - 返回車輛狀態
        if (strcmp(buffer, "GET_STATE") == 0) {
            // 準備車輛狀態數據
            float state[10];
            state[0] = car.x;
            // X位置
            state[1] = car.y;
            // Y位置
            state[2] = car.z;
            // Z位置
            state[3] = car.heading;
            // 方向角
            state[4] = car.speed;
            // 速度
            // 計算與賽道中心的橫向偏移
            float lateral = lateralOffsetFromCenter({car.x, car.y, car.z});
            state[5] = lateral;
            // 橫向偏移
            // 最近的賽道樣本點索引
            int nearestIdx = nearestSampleIndex({car.x, car.y, car.z});
            state[6] = static_cast<float>(nearestIdx);
            // 最近樣本點索引
            // 與賽道中心的距離
            state[7] = abs(lateral);
            // 與中心距離
            Vec3 carDir = {sinf(car.heading * M_PI / 180), 0, cosf(car.heading * M_PI / 180)};
            Vec3 trackDir;
            if (nearestIdx + 1 < static_cast<int>(trackSamples.size())) {
                trackDir = normalize(trackSamples[nearestIdx + 1] - trackSamples[nearestIdx]);
            } else {
                trackDir = normalize(trackSamples[0] - trackSamples[nearestIdx]);
            }
            // 計算夾角（弧度）
            float dotProduct = carDir.x * trackDir.x + carDir.z * trackDir.z;
            float angle = acosf(dotProduct) * 180.0f / M_PI;
            // 轉換為角度
            state[8] = angle;
            // 與賽道方向夾角
            // 擴展預留字段
            state[9] = 0.0f;
            // 發送狀態數據
            send(clientSocket, (const char*)state, sizeof(state), 0);
        } else if (strncmp(buffer, "ACTION", 6) == 0 && bytesReceived >= 7) {  // 處理ACTION命令 - 設置車輛按鍵狀態
            int action = static_cast<unsigned char>(buffer[6]);
            // 將動作ID轉換為按鍵狀態
            switch (action) {
                case 0:
                    // 無動作
                    car.kW = false;
                    car.kS = false;
                    car.kA = false;
                    car.kD = false;
                    break;
                case 1:
                    // 加速(W)
                    car.kW = true;
                    car.kS = false;
                    car.kA = false;
                    car.kD = false;
                    break;
                case 2:
                    // 煞車(S)
                    car.kW = false;
                    car.kS = true;
                    car.kA = false;
                    car.kD = false;
                    break;
                case 3:
                    // 左轉(A)
                    car.kW = false;
                    car.kS = false;
                    car.kA = true;
                    car.kD = false;
                    break;
                case 4:
                    // 右轉(D)
                    car.kW = false;
                    car.kS = false;
                    car.kA = false;
                    car.kD = true;
                    break;
                case 5:
                    // 加速+左轉(W+A)
                    car.kW = true;
                    car.kS = false;
                    car.kA = true;
                    car.kD = false;
                    break;
                case 6:
                    // 加速+右轉(W+D)
                    car.kW = true;
                    car.kS = false;
                    car.kA = false;
                    car.kD = true;
                    break;
                case 7:
                    // 煞車+左轉(S+A)
                    car.kW = false;
                    car.kS = true;
                    car.kA = true;
                    car.kD = false;
                    break;
                case 8:
                    // 煞車+右轉(S+D)
                    car.kW = false;
                    car.kS = true;
                    car.kA = false;
                    car.kD = true;
                    break;
            }
            send(clientSocket, "OK", 2, 0);
        } else if (strcmp(buffer, "RESET") == 0) {  // 處理RESET命令 - 重置賽車位置
            resetRace();
            // 返回確認
            send(clientSocket, "OK", 2, 0);
        } else if (strcmp(buffer, "LAP_INFO") == 0) {  // 處理LAP_INFO命令 - 返回圈數信息
            // 準備圈數信息
            int lapCount = lapP1;
            float lapTime = (float)raceTime;
            float travelDist = travelP1;
            int checkpointIdx = lastCheckpointP1;
            // 打包數據
            char responseData[16];
            memcpy(responseData, &lapCount, 4);
            memcpy(responseData + 4, &lapTime, 4);
            memcpy(responseData + 8, &travelDist, 4);
            memcpy(responseData + 12, &checkpointIdx, 4);
            // 發送圈數信息
            send(clientSocket, responseData, sizeof(responseData), 0);
        }
    } else if (bytesReceived == 0) {
        // 客戶端斷開連接
        cerr << "Python agent disconnected\n";
        clientConnected = false;
        closesocket(clientSocket);
        clientSocket = INVALID_SOCKET;
    } else {
        // recv返回SOCKET_ERROR，檢查是否只是非阻塞模式下的無數據
        int error = WSAGetLastError();
        if (error != WSAEWOULDBLOCK) {
            cerr << "recv failed: " << error << "\n";
            clientConnected = false;
            closesocket(clientSocket);
            clientSocket = INVALID_SOCKET;
        }
    }
}

// 關閉網絡連接

void cleanupNetwork() {
    if (clientSocket != INVALID_SOCKET) {
        closesocket(clientSocket);
    }
    if (serverSocket != INVALID_SOCKET) {
        closesocket(serverSocket);
    }
    WSACleanup();
}

// ------------------------------------------------------------
//  Entry
// ------------------------------------------------------------
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(winW, winH);
    glutCreateWindow("Speed Racing");
    atexit([] { setImeMode(true); });
    if (!initNetwork()) {
        cerr << "Failed to initialize network for AI mode\n";
    }
    initGL();
    prevMs = glutGet(GLUT_ELAPSED_TIME);

    glutDisplayFunc(displayCB);
    glutIdleFunc(updateCB);
    glutReshapeFunc(reshapeCB);
    glutSpecialFunc(specialKeyCB);
    glutKeyboardFunc(keyDown);
    glutKeyboardUpFunc(keyUp);
    glutMouseFunc(mouseCB);
    glutMotionFunc(motionCB);
    glutSpecialUpFunc(specialKeyUp);
    atexit(cleanupNetwork);
    glutMainLoop();

    return 0;
}