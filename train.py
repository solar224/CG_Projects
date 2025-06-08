from train_imports import *
SEED = 123

# --- Constants defined at the top of your script ---
ROAD_HALF_W_CONST = 15.0
SAND_HALF_W_CONST = 35.0
MAX_F_ROAD_CONST = 100.0
MAX_F_SAND_CONST = 60.0
MAX_F_GRASS_CONST = 15.0
TRACK_LEN = 1963.0
ROAD_W = ROAD_HALF_W_CONST
SAND_W = SAND_HALF_W_CONST

# --- Reward Function Constants (REVISED AGAIN) ---
# Goal 1: Track Completion
BONUS_LAP          = 30000.0 # 強力完成獎勵 (scaled: +300 -> QNet output clamped to +50)
K_PROGRESS         = 8.0     # 提高前進獎勵 (scaled: +0.08/meter)

# Goal 2: Speed (Target > 80, Max 125)
TARGET_SPEED_THRESHOLD = 80.0
K_HIGH_SPEED_BONUS = 20.0    # 維持高於80的獎勵 (scaled: up to +0.2)
K_SPEED_MAINTAIN   = 15.0    # 維持接近最高速的額外獎勵 (scaled: up to +0.15)
PEN_LOW_SPEED      = -5.0    # 速度 < 64 (80*0.8) 的懲罰 (scaled: -0.05) - 減輕此懲罰，鼓勵用idle減速
PEN_VERY_LOW_SPEED = -10.0   # 速度 < 40 (80*0.5) 的懲罰 (scaled: -0.1) - 同上

# Goal 3: Avoid Sand & Off-Track (CRITICAL for turns)
PEN_SAND_HIT       = -1800.0 # 極高懲罰 (scaled: -18.0)
PEN_SAND_EACH_FR   = -180.0  # 每幀在沙地上的高懲罰 (scaled: -1.8 per frame)
PEN_OFF_TRACK      = -2500.0 # 最高懲罰 (scaled: -25.0)

# Other Important Penalties
PEN_WRONG_WAY      = -40.0   # 倒車懲罰 (scaled: -0.4 for 1m backwards)

# --- Improving Turning Behavior ---
# Penalty for large angle with track when on road (discourages being sideways)
MAX_ANGLE_ON_ROAD_FOR_PENALTY = 20.0 # 度 (原25-30, 更嚴格)
PEN_LARGE_ANGLE_ON_ROAD = -10.0    # 加大懲罰 (scaled: -0.1)

# Penalty for excessive steering changes (wiggling), especially at speed
PEN_EXCESSIVE_STEERING_CHANGE = -5.0 # 加大懲罰 (scaled: -0.05)
MIN_SPEED_FOR_STEERING_PENALTY = MAX_F_ROAD_CONST * 0.25 # 速度門檻 (e.g., >31)
MAX_HEADING_CHANGE_BEFORE_PENALTY = 8.0 # 度/幀 (更嚴格)

# NEW: Penalty for hitting walls/barriers (inferred from sudden speed drop near edges)
PEN_WALL_HIT_IMPACT = -200.0 # (scaled: -2.0)

# --- Fine-tuning / Stability Rewards (Keep very small) ---
K_CENTER           = 0.1     # 極小的居中獎勵 (scaled: +0.001)
CENTER_POW         = 1.0
K_ALIGN            = 0.05    # 極小的對齊獎勵 (scaled: +0.0005)
ON_ROAD_BONUS      = 0.01    # 極微小的在路獎勵 (scaled: +0.0001)

ALPHA_SAND_ANGLE   = 1.3     # angle_k 影響力略微降低


# ACTION_TABLE and ACTION_SPACE should also be defined early if RacingEnvironment uses them in __init__
ACTION_TABLE = [0, 1, 5, 6]  # idle, W, WA, WD
ACTION_SPACE = len(ACTION_TABLE)
STEER_LEFT_ACTIONS  = {2}  # WA 的索引是 2 (對應 ACTION_TABLE[2] 即遊戲動作 5)
STEER_RIGHT_ACTIONS = {3}  # WD 的索引是 3 (對應 ACTION_TABLE[3] 即遊戲動作 6)
STRAIGHT_ACTIONS    = {0, 1} # idle 和 W 的索引


# 訊號處理函數，用於處理程式終止訊號
def signal_handler(sig, frame):
    print('\n捕獲到終止訊號，正在退出...')
    # 記錄中斷事件到logs.csv
    log_to_csv(
        time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        runtime=time.time() - start_time if 'start_time' in globals() else 0,
        reason=f"程式被終止訊號中斷 (訊號: {sig})"
    )
    # 檢查是否啟用了郵件通知
    if 'email_config' in globals() and email_config['enabled'] and email_config['notify_errors']:
        subject = "賽車訓練通知: 訓練被終止"
        message = (f"訓練程式被手動終止或系統訊號中斷!\n\n"
                  f"終止時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"終止訊號: {sig}\n")
        # 嘗試發送通知
        try:
            send_email_notification(subject=subject, message=message)
        except Exception as e:
            print(f"發送終止通知時出錯: {e}")
    print("程式已終止")
    sys.exit(0)

# 將訓練中斷或完成的記錄存到logs.csv
def log_to_csv(time, runtime, reason):
    """
    參數:
        time (str): 中斷或完成的時間 (YYYY-MM-DD HH:MM:SS)
        runtime (float): 程式運行總秒數
        reason (str): 中斷原因或完成說明
    """
    # 將 runtime 轉成 HH:MM:SS
    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    runtime_formatted = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    logfile = 'logs.csv'
    file_exists = os.path.isfile(logfile) and os.path.getsize(logfile) > 0

    with open(logfile, 'a', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['時間', '運作總時間', '中斷原因']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 第一次寫入時先加標題列
        if not file_exists:
            writer.writeheader()

        # 寫入本次紀錄
        writer.writerow({
            '時間': time,
            '運作總時間': runtime_formatted,
            '中斷原因': reason
        })

    print(f"已記錄事件到 {logfile}: {reason}")
# 設定隨機種子以確保可重現性
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# 啟用 autograd 異常檢測，用於偵測 NaN
torch.autograd.set_detect_anomaly(True)

# 檢查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

class SingleHeadDQN(nn.Module):
    def __init__(self, state_size, action_size=ACTION_SPACE):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.LayerNorm(256)
        )
        # Dueling
        self.value = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.adv   = nn.Sequential(nn.Linear(256, 128), nn.SiLU(),
                                   nn.Linear(128, action_size, bias=False))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.feature(x)
        v = self.value(z)
        a = self.adv(z)
        return v + (a - a.mean(dim=1, keepdim=True))   # (B,4)

# 支持雙輸出Agent的經驗回放記憶體
class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)

    
    def add(self, tup): self.buffer.append(tup)   # (s,a,r,s',d)
    
    def sample(self, batch):
        idx = random.sample(range(len(self.buffer)), batch)
        s  = np.array([self.buffer[i][0] for i in idx])
        a  = np.array([[self.buffer[i][1]] for i in idx])
        r  = np.array([[self.buffer[i][2]] for i in idx])
        s2 = np.array([self.buffer[i][3] for i in idx])
        d  = np.array([[self.buffer[i][4]] for i in idx])
        return s,a,r,s2,d

    def __len__(self): return len(self.buffer)
    

# 新的DualOutputDQNAgent，使用分離的轉向和油門控制
class DualOutputDQNAgent:
    def __init__(self, state_size, 
                 lr=3e-5, gamma=0.99, epsilon=1.0, epsilon_decay=0.9975, epsilon_min=0.05,
                 buffer_size=200000, batch_size=256, update_target_freq=7500,
                 epsilon_decay_mode='exponential', decay_steps=20000,
                 huber_delta=5.0, grad_clip_norm=5.0, weight_decay=1e-4):
        self.state_size = state_size
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰減（指數衰減時使用）
        self.epsilon_min = epsilon_min  # 最小探索率
        self.batch_size = batch_size  # 批次大小
        self.update_target_freq = update_target_freq  # 目標網路更新頻率
        self.train_step = 0  # 訓練步數計數器
        self.huber_delta = huber_delta  # Huber損失的delta值 # 將由 args.huber_delta 傳入
        self.grad_clip_norm = grad_clip_norm  # 梯度裁剪範數 # 將由 args.grad_clip_norm 傳入
        self.weight_decay = weight_decay  # AdamW權重衰減參數
        self.action_size = ACTION_SPACE # 使用全域 ACTION_SPACE
        # Epsilon衰減模式設置
        self.epsilon_decay_mode = epsilon_decay_mode  # 'exponential' 或 'linear'
        self.decay_steps = decay_steps  # 線性衰減的總步數 # 將由 args.decay_steps 傳入
        
        # 前進預訓練模式
        self.forward_pretraining_mode = False
        
        # 計算線性衰減的參數
        if self.epsilon_decay_mode == 'linear' and self.decay_steps is not None and self.decay_steps > 0:
            self.epsilon_decay_per_step = (self.epsilon - self.epsilon_min) / self.decay_steps
        else:
            self.epsilon_decay_per_step = 0
        # 主要網路和目標網路
        self.model = SingleHeadDQN(state_size, action_size=ACTION_SPACE).to(device)
        self.target_model = SingleHeadDQN(state_size, action_size=ACTION_SPACE).to(device)
        self.update_target_model()  # 初始化目標網路權重
        # 使用AdamW優化器，添加權重衰減
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay) # lr 將由 args.lr 傳入
        # 使用Huber損失(SmoothL1Loss)替代MSE，更好地處理異常值，減少數值爆炸 
        self.loss_fn = nn.SmoothL1Loss() # 使用 self.huber_delta
        # 經驗回放記憶體
        self.memory = ReplayBuffer(buffer_size)
        self.rewards_history = []  # 記錄每個episode的總獎勵
        self.episode_count = 0  # 記錄當前訓練到的episode數
        # 追踪指標
        self.q_values_history = []  # 記錄每個episode的平均Q值
        self.current_episode_q_values = []  # 當前episode的所有Q值
        # 狀態標準化參數
        self.state_means = np.zeros(state_size)
        self.state_stds = np.ones(state_size)
        self.state_normalization_initialized = False
        self.normalization_samples = []
        # 記錄前一個轉向動作，用於轉向平滑獎勵計算
        self.prev_steering_action = None
        # 預分配狀態緩衝區以避免反覆創建新數組
        self.scaled_state_buffer = np.zeros(state_size, dtype=np.float32)

    def normalize_state(self, state):
        # 此方法現在可以直接使用在檔案頂部定義的全域常數
        scaled = np.zeros_like(state, dtype=np.float32)
        for i in range(len(state)):
            if i == 0 or i == 2:  # x, z coordinates
                scaled[i] = state[i] / (500.0 + 1e-6)
            elif i == 1:  # y coordinate (height)
                scaled[i] = state[i] / (100.0 + 1e-6)
            elif i == 3:  # heading (angle)
                scaled[i] = state[i] / (180.0 + 1e-6)
            elif i == 4:  # speed
                scaled[i] = state[i] / (MAX_F_ROAD_CONST + 1e-6) # 使用全域 MAX_F_ROAD_CONST
            elif i == 5:  # lateral_offset
                scaled[i] = state[i] / (ROAD_HALF_W_CONST + 1e-6) # 使用全域 ROAD_HALF_W_CONST
            elif i == 6:  # nearest_sample_idx
                scaled[i] = state[i] / 1963.0 
            elif i == 7:  # distance_from_center
                scaled[i] = state[i] / (SAND_HALF_W_CONST + 1e-6) # 使用全域 SAND_HALF_W_CONST
            elif i == 8:  # angle_with_track
                scaled[i] = state[i] / (90.0 + 1e-6)
        scaled = np.nan_to_num(scaled, nan=0.0)
        return scaled


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, training=True):
        # 標準化狀態
        s = torch.as_tensor(self.normalize_state(state), dtype=torch.float32,
                            device=device).unsqueeze(0)      # === 一次 forward，先拿到 Q ===
        with torch.no_grad():
            q = self.model(s)              # shape (1,6)
            
        # --- ε-greedy / 強迫加速 ---
        if training and np.random.rand() < self.epsilon:
            action_id = np.random.randint(self.action_size)
        else:
            action_id = torch.argmax(q, dim=1).item()
            
# 如果在前進預訓練模式下，強制使用直線動作
        if self.forward_pretraining_mode and training: # Ensure 'training' flag is also checked
            p = np.random.rand()
            if p < 0.6:       # 60% W (increased W probability)
                action_id = 1 # ACTION_TABLE[1] is 'W'
            elif p < 0.8:     # 20% WA
                action_id = 2 # ACTION_TABLE[2] is 'WA' (value 5)
            else:             # 20% WD
                action_id = 3 # ACTION_TABLE[3] is 'WD' (value 6)

        # 統計
        if torch.isfinite(q).all():
            self.current_episode_q_values.append(q.max().item())

        return ACTION_TABLE[action_id], action_id      # 回傳 (原始遊戲動作, 內部 id)
    
    def remember(self, state, action_id, reward, next_state, done):
        exp = (self.normalize_state(state), action_id, reward,
            self.normalize_state(next_state), done)
        self.memory.add(exp)
    
    def train(self):
        # 若記憶體中的樣本不足，則不進行學習
        if len(self.memory) < self.batch_size:
            return 0
        
        # 從記憶體中隨機抽樣一批經驗，已經轉換為numpy數組
        s,a,r,s2,d = self.memory.sample(self.batch_size)
        s   = torch.as_tensor(s,  dtype=torch.float32, device=device)
        a   = torch.as_tensor(a,  dtype=torch.int64,  device=device)
        r_tensor   = torch.as_tensor(r,  dtype=torch.float32, device=device)
        s2  = torch.as_tensor(s2, dtype=torch.float32, device=device)
        d   = torch.as_tensor(d,  dtype=torch.float32, device=device)

        q_pred = self.model(s).clamp_(-50, 50).gather(1, a)      # [-50, 50] 夠用了
        
        with torch.no_grad():
            # 注意 next_q 也要 clamp
            next_a  = self.model(s2).argmax(1, keepdim=True)
            next_q  = self.target_model(s2).clamp_(-50, 50).gather(1, next_a)
            tgt_q = r_tensor / 100.0 + (1-d) * self.gamma * next_q


        # ---------- ② loss ----------
        loss = self.loss_fn(q_pred, tgt_q)

        # ---------- ③ backward — 梯度安全閥 ----------
        self.optimizer.zero_grad()
        loss.backward()

        # 先做 NaN/Inf 檢查
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print("⚠️ grad contains NaN/Inf – skip update")
                return loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # 線性衰減epsilon（以步數為單位）
        if self.epsilon_decay_mode == 'linear' and self.train_step < self.decay_steps and self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_per_step
            self.epsilon = max(self.epsilon, self.epsilon_min)
        
        # 更新目標網路
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.update_target_model()
            print(f"目標網路已更新，訓練步數: {self.train_step}")
        
        return loss.item()
    
    def decay_epsilon(self):
        """根據設定的衰減模式調整epsilon值"""
        self.episode_count += 1
        
        if self.epsilon_decay_mode == 'exponential':
            # 指數衰減: epsilon = epsilon * decay_rate，但不低於最小值
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)  # 確保不會低於最小值
        
        elif self.epsilon_decay_mode == 'linear' and self.decay_steps is not None:
            # 在訓練時每步衰減，而非每個回合
            pass  # 在train方法中執行線性衰減
    
    def reset_episode_stats(self):
        """重置每個episode的統計數據"""
        self.current_episode_q_values = []
    
    def get_episode_stats(self):
        """
        回傳當前 episode 的統計數據，保證包含 avg_q
        """
        # ---- 1. 過濾 NaN ----
        valid_q_vals = [q for q in self.current_episode_q_values if not math.isnan(q)]
        avg_q       = np.mean(valid_q_vals) if valid_q_vals else 0.0
        # ---- 3. 存歷史 (如果需要視覺化) ----
        self.q_values_history.append(avg_q)

        # ---- 4. 回傳 ----
        return {'avg_q': avg_q}
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'rewards_history': self.rewards_history
        }, filepath)
        print(f"模型已保存至 {filepath}")
    
    def load(self, filepath):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.train_step = checkpoint.get('train_step', 0)
            self.rewards_history = checkpoint.get('rewards_history', [])
        else:
            print(f"找不到模型檔案: {filepath}")

# 遊戲環境接口

# 電子郵件通知功能
def send_email_notification(subject, message, recipient_email=None, sender_email=None, password=None, attachments=None):
    """
    發送電子郵件通知
    
    參數:
    - subject: 郵件主題
    - message: 郵件內容
    - recipient_email: 收件人郵箱，如果為None則使用全局配置
    - sender_email: 發件人郵箱，如果為None則使用全局配置
    - password: 發件人密碼，如果為None則使用全局配置
    - attachments: 附件列表，格式為 [(filename, filepath), ...]
    """
    # 使用全局配置或參數值
    if recipient_email is None and 'email_config' in globals() and email_config['recipient']:
        recipient_email = email_config['recipient']
    elif recipient_email is None:
        recipient_email = "C110110157@gmail.com"  # 默認收件人
    
    if sender_email is None and 'email_config' in globals() and email_config['sender']:
        sender_email = email_config['sender']
    elif sender_email is None:
        sender_email = "training.notification@gmail.com"  # 默認發件人
    
    if password is None and 'email_config' in globals() and email_config['password']:
        password = email_config['password']
    elif password is None:
        # 如果沒有提供密碼，則無法發送郵件
        print("錯誤: 未提供郵件發送密碼，無法發送通知")
        return False
    
    try:
        # 創建郵件
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # 添加機器識別訊息
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        system_info = f"Hostname: {hostname}\nIP: {ip_address}\n"
        
        # 添加正文
        body = system_info + "\n" + message
        msg.attach(MIMEText(body, 'plain'))
        
        # 添加附件
        if attachments:
            for filename, filepath in attachments:
                if os.path.exists(filepath):
                    with open(filepath, "rb", encoding='utf-8-sig') as f:
                        part = MIMEApplication(f.read(), Name=filename)
                    part['Content-Disposition'] = f'attachment; filename="{filename}"'
                    msg.attach(part)
        
        # 連接到SMTP服務器並發送郵件
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # 啟用TLS加密
        
        try:
            server.login(sender_email, password)
        except smtplib.SMTPAuthenticationError as auth_error:
            # Gmail認證錯誤
            print("\n===== Gmail認證錯誤 =====")
            print("由於Google的安全政策，您需要使用「應用程式密碼」而不是普通的Gmail密碼。")
            print("請按照以下步驟設置應用程式密碼：")
            print("1. 前往 https://myaccount.google.com/security")
            print("2. 確保您已開啟「兩步驟驗證」")
            print("3. 點擊「應用程式密碼」，選擇「其他」並輸入一個名稱(例如「訓練程式」)")
            print("4. 複製生成的16位字符密碼")
            print("5. 使用此密碼作為--email_password參數")
            print("\n錯誤詳情:", str(auth_error))
            print("============")
            server.quit()
            return False
            
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        print(f"電子郵件通知已發送至 {recipient_email}")
        return True
    except smtplib.SMTPException as e:
        print(f"發送電子郵件時出錯 (SMTP錯誤): {str(e)}")
        return False
    except socket.gaierror:
        print(f"發送電子郵件時出錯: 無法連接到SMTP服務器，請檢查網絡連接")
        return False
    except Exception as e:
        print(f"發送電子郵件時出錯: {str(e)}")
        return False

class RacingEnvironment:
    def __init__(self, exe_path="racing.exe", debug=False):
        self.total_samples = None     #  ←  先留空
        self.exe_path = exe_path
        self.state_size = 9  # 實際遊戲返回的狀態維度: [x, y, z, heading, speed, lateral_offset, nearest_sample_idx, distance_from_center, angle_with_track]

        self.action_size = ACTION_SPACE # Uses global ACTION_SPACE
        # self.model = SingleHeadDQN(self.state_size, self.action_size).to(device) # Consider moving model creation outside env

        # self.model = SingleHeadDQN(self.state_size, self.action_size).to(device)
        self.process = None
        self.lap_time = 0
        self.travel_distance = 0
        self.last_checkpoint = 0
        self.lap_count = 0
        self.game_interface = GameInterface() # Assuming GameInterface is defined/imported
        self.prev_distance = 0  # 用於計算進度獎勵
        self.timeout = 180  # 單圈超時時間（秒）- 增加容許時間
        self.start_time = 0  # 記錄開始時間
        self.episode_count = 0  # 回合計數，用於動態調整獎勵
        # 追踪上一個動作，用於計算轉向平滑度獎勵
        self.prev_steering_action = None
        # 連續轉向計數器 (用於轉向過久懲罰)
        self.continuous_steering_count = 0
        self.sand_stuck_counter = 0   # 在沙地且沒速度的連續幀數
        # 新增統計數據
        self.crash_count = 0  # 撞車次數
        self.speed_history = []  # 速度歷史記錄
        self.distance_history = []  # 距離中心的歷史記錄
        # 新增卡住檢測
        self.stuck_counter = 0  # 卡住計數器
        self.stuck_threshold = 300    # 卡住閾值（幀數）
        self.min_progress_threshold = 0.05  # 最小進度閾值
        
        # 新增里程碑追踪
        self.last_milestone = 0  # 最後達到的里程碑 (0-20) - 因為改為每5%一個里程碑
        self.last_rewarded_lap = 0  # 最後獎勵的圈數
        self.low_speed_time = 0.0  # 低速持續時間計數器
        self.steps_in_episode = 0  # 當前回合的步數
        
        self.debug = debug
        self.debug_rewards = False
        self.debug_interval = 20  # 每隔20步輸出一次訊息
        self._in_sand_prev = False

        self.reset_episode_stats()
    
    def _start_game(self, mode="MODE_AI"):
        """啟動遊戲進程並設置為AI模式"""
        try:
            # 開啟新的子進程執行遊戲，使用start_new_session=True確保即使父進程被kill也不會留下殭屍進程
            if os.name == 'nt':  # Windows
                self.process = subprocess.Popen([self.exe_path, mode], 
                                          stdin=subprocess.PIPE, 
                                          stdout=subprocess.PIPE,
                                          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:  # Linux/Mac
                self.process = subprocess.Popen([self.exe_path, mode], 
                                          stdin=subprocess.PIPE, 
                                          stdout=subprocess.PIPE,
                                          start_new_session=True)
            time.sleep(2)  # 等待遊戲啟動
            
            # 嘗試建立連接
            connected = self.game_interface.connect()
            if not connected:
                print("無法連接到遊戲，將嘗試關閉並重新啟動")
                if self.process:
                    self.process.terminate()
                    time.sleep(1)
                return False
            
            return True
        except Exception as e:
            print(f"啟動遊戲失敗: {e}")
            return False
    
    def reset(self):
        """重置環境，重新開始賽車"""
        # 更新回合計數
        self.episode_count += 1
        
        # 如果已有連接，先嘗試發送重置命令
        if self.game_interface.connected:
            reset_success = self.game_interface.reset_game()
            if reset_success:
                time.sleep(1.5)  # 等待遊戲重置
                # 重置內部狀態
                self.lap_time = 0
                self.travel_distance = 0
                self.last_checkpoint = 0
                self.lap_count = 0
                self.prev_distance = 0  # 確保prev_distance重置為0
                self.start_time = time.time()
                self.stuck_counter = 0  # 重置卡住計數器
                self.last_milestone = 0  # 重置里程碑計數器
                self.last_rewarded_lap = 0  # 重置圈數獎勵計數器
                self.low_speed_time = 0.0  # 重置低速計時器
                self.steps_in_episode = 0  # 重置回合步數
                self.prev_steering_action = None  # 重置前一個轉向動作
                self.continuous_steering_count = 0  # 重置連續轉向計數器
                self.reset_episode_stats()  # 重置統計數據
                
                # 每10回合輸出一次訊息
                if self.debug and self.episode_count % 10 == 0:
                    self.debug_rewards = True
                else:
                    self.debug_rewards = False
                    
                return self.game_interface.get_state()
        
        # 如果沒有連接或重置失敗，則關閉舊進程並啟動新遊戲
        if self.process:
            self.process.terminate()
            time.sleep(1)
            
        success = self._start_game()
        if not success:
            raise Exception("無法啟動遊戲環境")
        
        # 重置內部狀態
        self.lap_time = 0
        self.travel_distance = 0
        self.last_checkpoint = 0
        self.lap_count = 0
        self.prev_distance = 0  # 確保prev_distance重置為0
        self.start_time = time.time()
        self.stuck_counter = 0  # 重置卡住計數器
        self.last_milestone = 0  # 重置里程碑計數器
        self.last_rewarded_lap = 0  # 重置圈數獎勵計數器 
        self.low_speed_time = 0.0  # 重置低速計時器
        self.steps_in_episode = 0  # 重置回合步數
        self.prev_steering_action = None  # 重置前一個轉向動作
        self.continuous_steering_count = 0  # 重置連續轉向計數器
        self.reset_episode_stats()  # 重置統計數據
        
        # 獲取初始狀態
        time.sleep(1.5)  # 給遊戲一點時間初始化
        return self.game_interface.get_state()

    def _calculate_reward(self, s_prev, s_next, done):
        v_prev = np.clip(s_prev[4], 0., MAX_F_ROAD_CONST) # Speed from previous state
        v_next = np.clip(s_next[4], 0., MAX_F_ROAD_CONST) # Current speed (after action)
        
        d_center_next = abs(s_next[7])
        on_road_next = d_center_next <= ROAD_HALF_W_CONST
        on_sand_next = (ROAD_HALF_W_CONST < d_center_next <= SAND_HALF_W_CONST)
        off_track_next = d_center_next > SAND_HALF_W_CONST

        theta_next = abs(s_next[8]) 
        angle_k_next = 1.0 + (theta_next / (90.0 + 1e-6)) ** ALPHA_SAND_ANGLE

        heading_prev = s_prev[3]
        heading_next = s_next[3]
        heading_diff = heading_next - heading_prev
        if heading_diff > 180: heading_diff -= 360
        elif heading_diff < -180: heading_diff += 360
        abs_heading_change_per_step = abs(heading_diff)

        r = 0.0

        # --- 1. Track Completion & Progress ---
        dp = self.travel_distance - self.prev_distance 
        if dp > 0.01: 
            r += K_PROGRESS * dp 
            # Reward for maintaining angle while progressing (good for exiting turns)
            if theta_next < MAX_ANGLE_ON_ROAD_FOR_PENALTY * 0.75: # If well aligned
                 r += dp * K_ALIGN * 2.0 # Small bonus for progressing straight
        elif dp < -0.005: 
            r += PEN_WRONG_WAY * abs(dp) 

        if self.lap_count > self.last_rewarded_lap:
            r += BONUS_LAP 
            self.last_rewarded_lap = self.lap_count
            print(f"🎉 Lap {self.lap_count} completed! Bonus: {BONUS_LAP}")

        # --- 2. Speed Management ---
        if on_road_next:
            if v_next > TARGET_SPEED_THRESHOLD:
                speed_excess_ratio = (v_next - TARGET_SPEED_THRESHOLD) / (MAX_F_ROAD_CONST - TARGET_SPEED_THRESHOLD + 1e-6)
                r += K_HIGH_SPEED_BONUS * np.clip(speed_excess_ratio, 0.0, 1.0)
                if v_next > MAX_F_ROAD_CONST * 0.9:
                    r += K_SPEED_MAINTAIN * (v_next / (MAX_F_ROAD_CONST + 1e-6))
            elif v_next < (TARGET_SPEED_THRESHOLD * 0.5):
                r += PEN_VERY_LOW_SPEED 
            elif v_next < (TARGET_SPEED_THRESHOLD * 0.8):
                r += PEN_LOW_SPEED
        else: # Off-road - harsher speed penalty
            if v_next > MAX_F_SAND_CONST * 0.1:
                 r -= K_HIGH_SPEED_BONUS * 1.2 * (v_next / (MAX_F_SAND_CONST + 1e-6))

        # --- 3. Avoid Sand & Off-Track ---
        if on_sand_next:
            if not self._in_sand_prev: # Just hit sand (self._in_sand_prev was updated at end of previous step)
                r += PEN_SAND_HIT 
                print(f"💥 Hit Sand! Penalty: {PEN_SAND_HIT:.2f}")
            r += PEN_SAND_EACH_FR * angle_k_next
        
        if off_track_next:
            r += PEN_OFF_TRACK * angle_k_next
            print(f"🏞️ Off Track! Penalty: {PEN_OFF_TRACK * angle_k_next:.2f}")

        # --- 4. Improving Turning Behavior ---
        if on_road_next:
            # Penalize large angle with track (being sideways)
            if theta_next > MAX_ANGLE_ON_ROAD_FOR_PENALTY:
                penalty_factor = ((theta_next - MAX_ANGLE_ON_ROAD_FOR_PENALTY) / 
                                  (90.0 - MAX_ANGLE_ON_ROAD_FOR_PENALTY + 1e-6))
                if d_center_next > ROAD_HALF_W_CONST * 0.7 or dp < 0.02: # Wider margin for being off-center, or very slow progress
                    penalty_factor *= 2.0 # Stronger penalty if also drifting wide or stuck
                r += PEN_LARGE_ANGLE_ON_ROAD * penalty_factor
            
            # Penalize excessive steering change (wiggling) if speed is somewhat high
            if v_next > MIN_SPEED_FOR_STEERING_PENALTY and abs_heading_change_per_step > MAX_HEADING_CHANGE_BEFORE_PENALTY:
                r += PEN_EXCESSIVE_STEERING_CHANGE * (abs_heading_change_per_step / (30.0 + 1e-6)) # Normalized by a smaller divisor

            # NEW: Penalty for sudden speed drop near edges (potential wall hit)
            # This is an approximation for hitting a wall/barrier
            is_near_edge = (d_center_next > ROAD_HALF_W_CONST * 0.8) # If close to edge of road
            sudden_speed_drop = (v_prev > v_next * 1.5) and (v_prev > MIN_SPEED_FOR_STEERING_PENALTY) # Significant speed drop
            if on_road_next and is_near_edge and sudden_speed_drop and dp < 0.1 : # if also not making progress
                r += PEN_WALL_HIT_IMPACT
                print(f"🧱 Possible Wall Hit! Penalty: {PEN_WALL_HIT_IMPACT}")
        
        # --- 5. Fine-tuning / Stability Rewards (Keep very small) ---
        if on_road_next:
            r_center = K_CENTER * (1.0 - (d_center_next / (ROAD_HALF_W_CONST + 1e-6)) ** CENTER_POW)
            r += r_center
            r += K_ALIGN * (1.0 - theta_next / (90.0 + 1e-6))
            r += ON_ROAD_BONUS
        
        self._in_sand_prev = on_sand_next or off_track_next
        self.prev_distance = self.travel_distance
        return r


    def _sanitize_state(self, s):
        """把 NaN/Inf 轉成有限值，避免訓練整段 crash。"""
        # 把 NaN → 0，+Inf → 大數，-Inf → 小數
        s = np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)
        # 限制每個欄位的合理範圍，防止極端值害死網路
        # x, z 座標
        s[0] = np.clip(s[0], -1000, 1000)
        s[2] = np.clip(s[2], -1000, 1000)
        # speed
        s[4] = np.clip(s[4], -1.5*MAX_F_SAND_CONST, 1.5*MAX_F_ROAD_CONST)
        # 與中心距離
        s[7] = np.clip(s[7], -SAND_HALF_W_CONST*1.5,  SAND_HALF_W_CONST*1.5)
        # …視需要再加欄位
        return s

    def step(self, action):
        self.steps_in_episode += 1
        pre_action_travel_distance = self.travel_distance
        
        # s_prev for _calculate_reward is current_state_in_step
        s_prev_for_reward = self.game_interface.get_state() # This is the state before the action is taken

        self.game_interface.send_action(action)
        time.sleep(0.04)

        next_state_raw = self.game_interface.get_state() # This is s_next
        next_state = self._sanitize_state(next_state_raw)

        # Update lap info after getting next_state
        self.lap_count, self.lap_time, self.travel_distance, self.last_checkpoint = self.game_interface.get_lap_info()

        if self.total_samples is None or self.last_checkpoint > self.total_samples:
            self.total_samples = self.last_checkpoint

        self.speed_history.append(next_state[4])
        self.distance_history.append(next_state[7])

        # Pass current_state_in_step as s_prev
        reward = self._calculate_reward(s_prev_for_reward, next_state, False)
        
        done = False
        crashed = False
        stuck = False

        if self.lap_count >= 1:
            done = True
            print(f"🏁 Episode finished: Lap Completed! Raw Reward: {reward:.2f}")
        
        if not done:
            current_step_progress = self.travel_distance - pre_action_travel_distance
            if current_step_progress < self.min_progress_threshold and next_state[4] < 10.0:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            
            if self.stuck_counter >= self.stuck_threshold:
                done = True
                stuck = True
                reward += -800.0  # Further increase override penalty for being properly stuck
                print(f"🚫 Agent Stuck! Reward overridden to: {reward:.2f}")

        if not done:
            is_off_main_sand_area = abs(next_state[7]) > SAND_HALF_W_CONST # Use self.
            is_timeout = (time.time() - self.start_time) > self.timeout
            if is_off_main_sand_area or is_timeout:
                done = True
                crashed = True
                self.crash_count += 1
                reason = "OffTrack" if is_off_main_sand_area else "Timeout"
                reward += -1000.0 # Further increase override penalty
                print(f"💥 Agent Crashed ({reason})! Reward overridden to: {reward:.2f}")
        
        if not done:
            if not np.isfinite(next_state).all():
                bad_indices = np.where(np.isnan(next_state) | np.isinf(next_state))[0]
                print(f"[DEBUG] Non-finite state: indices={bad_indices}, vals={next_state[bad_indices]}")
                done = True
                crashed = True
                reward += -700.0
                print(f"💣 NaN State! Reward overridden to: {reward:.2f}")

        if not done:
            is_on_sand_for_stuck_check = (ROAD_HALF_W_CONST < abs(next_state[7]) <= SAND_HALF_W_CONST) # Use self.
            if is_on_sand_for_stuck_check and next_state[4] < 5.0:
                self.sand_stuck_counter +=1
            else:
                self.sand_stuck_counter = 0
            
            if self.sand_stuck_counter >= 60: 
                done = True
                crashed = True
                reward += -400.0 
                print(f"⏳ Stuck in Sand! Additional penalty. Reward now: {reward:.2f}")
        
        assert np.isfinite(next_state).all(), f"State NaN/Inf: {next_state}"
        assert action in ACTION_TABLE, f"Invalid action: {action}" # ACTION_TABLE is global
        assert math.isfinite(reward), f"Reward NaN/Inf: {reward}"

        return next_state, reward, done, {
            'lap_count': self.lap_count, 'lap_time': self.lap_time,
            'travel_distance': self.travel_distance, 'checkpoint': self.last_checkpoint,
            'crashed': crashed, 'stuck': stuck
        }

    def reset_episode_stats(self):
        """重置每個episode的統計數據"""
        self.speed_history = []
        self.distance_history = []
        self.sand_stuck_counter = 0
    def get_episode_stats(self):
        """獲取當前episode的統計數據"""
        # 車速分布統計
        speeds = np.abs(np.array(self.speed_history))
        speed_stats = {
            'mean_speed': np.mean(speeds) if len(speeds) > 0 else 0,
            'max_speed': np.max(speeds) if len(speeds) > 0 else 0,
            'min_speed': np.min(speeds) if len(speeds) > 0 else 0,
            'speed_std': np.std(speeds) if len(speeds) > 0 else 0,
        }
        bins = [0, 30, 60, 90, float('inf')]
        hist, _ = np.histogram(speeds, bins=bins)
        speed_dist = hist / hist.sum() if hist.sum() else np.zeros(4)

        return {
            'crash_count'      : self.crash_count,
            'speed_stats'      : {
                k: float(v) for k, v in speed_stats.items()   # 轉成 float
            },
            'speed_distribution': speed_dist,
        }
        
    def close(self):
        """關閉環境和遊戲進程"""
        if self.game_interface.connected:
            self.game_interface.disconnect()
            
        if self.process:
            try:
                # 嘗試正常終止進程
                self.process.terminate()
                
                # 給進程一些時間正常關閉
                for _ in range(5):  # 嘗試最多5次，每次等待0.5秒
                    if self.process.poll() is not None:  # 如果進程已退出
                        break
                    time.sleep(0.5)
                
                # 如果進程還沒退出，強制關閉
                if self.process.poll() is None:
                    print("遊戲進程未正常終止，嘗試強制關閉")
                    if os.name == 'nt':  # Windows
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)], 
                                      stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    else:  # Linux/Mac
                        import signal
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except Exception as e:
                print(f"關閉遊戲進程時出錯: {e}")
            finally:
                self.process = None

# 訓練函數
def train_agent(agent, env, num_episodes, max_steps_per_episode=3000, 
                save_freq=100, model_dir="models", log_interval=1,
                warmup_steps=5000, forward_pretraining=True, pretraining_episodes=30,
                turn_training_episodes=10):
    
    os.makedirs(model_dir, exist_ok=True)
    
    # 創建logs目錄
    log_dir = os.path.join(model_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 初始化CSV記錄器
    csv_log_path = os.path.join(log_dir, f"training_log_{agent.__class__.__name__}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    csv_fields = ['episode', 'steps', 'total_reward', 'avg_reward', 'epsilon', 'loss', 'mean_q', 'mean_speed', 
                 'crash_count', 'completed']
    
    
    # 創建CSV文件並寫入欄位名稱
    with open(csv_log_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer_csv = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer_csv.writeheader()
    
    total_rewards = []
    avg_rewards = []
    
    # 新增監控指標
    mean_q_values = []
    crash_counts = []
    completed_laps = []
    mean_speeds = []
    
    # 新增完成百分比追踪
    completion_percentages = []
    # 速度分布累積數據 (0-30, 30-60, 60-90, 90+)
    speed_distribution = np.zeros(4)
    speed_dist_counts = 0
    
    
    # 儲存原始epsilon設定，用於後續調整
    original_epsilon_min = agent.epsilon_min
    
    # 進行Replay Buffer預熱
    if warmup_steps > 0:
        print(f"開始Replay Buffer預熱，執行 {warmup_steps} 步隨機動作...")
        state = env.reset()
        original_epsilon = agent.epsilon  # 保存原始epsilon
        agent.epsilon = 1.0  # 確保完全隨機
        
        for step in range(warmup_steps):
            # 選擇隨機動作
            action_game, action_id = agent.act(state)

            
            # 執行動作
            next_state, reward, done, info = env.step(action_game)
            
            # 記憶經驗，但不訓練
            agent.remember(state, action_id, reward, next_state, done)
            
            # 更新狀態
            state = next_state
            
            if done:
                state = env.reset()
                print(f"預熱重置: 步數 {step+1}/{warmup_steps}")
        
        # 恢復原始epsilon值
        agent.epsilon = original_epsilon
        print("Replay Buffer預熱完成")
    
    # 前進動作預訓練 (只允許向前/靜止動作)
    if forward_pretraining and pretraining_episodes > 0:
        print(f"開始前進動作預訓練，執行 {pretraining_episodes} 個回合...")
        
        # 保存原始epsilon值用於後續恢復
        original_epsilon = agent.epsilon
        
        # 設置較高的探索率以鼓勵嘗試前進動作
        agent.epsilon = 0.7
        
        # 啟用前進預訓練模式
        agent.forward_pretraining_mode = True
        
        for episode in range(1, pretraining_episodes + 1):
            # 重置環境
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                # 選擇動作 - 只會是前進(W)或靜止(idle)
                action_game, action_id = agent.act(state, training=True)
                
                # 執行動作
                next_state, reward, done, info = env.step(action_game)
                
                # 記憶經驗
                agent.remember(state, action_id, reward, next_state, done)
                
                # 訓練網絡
                agent.train()
                
                # 更新狀態
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            print(f"前進預訓練 - Episode {episode}/{pretraining_episodes}, 步數: {steps}, 獎勵: {total_reward:.2f}")
        
        # 恢復原始epsilon值並關閉前進預訓練模式
        agent.epsilon = original_epsilon
        agent.forward_pretraining_mode = False
        print("前進動作預訓練完成")
    
    # 主要訓練階段
    print(f"開始主要訓練階段，執行 {num_episodes} 個回合...")
    
    # 重置環境狀態
    env.reset_episode_stats()
    agent.reset_episode_stats()
    
    # 記錄訓練開始時間
    start_training_time = time.time()
    
    # 圈數完成通知計數
    lap_completion_count = 0
    
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()
        
        # 重置環境，獲取初始狀態
        state = env.reset()
        
        # 檢查初始狀態是否有效（避免負速度或極端角度方向的情況）
        invalid_state_counter = 0
        max_reset_attempts = 3
        
        while (state[4] < 0 or abs(state[8]) > 120) and invalid_state_counter < max_reset_attempts:
            print(f"檢測到無效初始狀態: 速度={state[4]:.2f}, 角度={state[8]:.2f}. 嘗試重置...")
            state = env.reset()
            invalid_state_counter += 1
            
        if invalid_state_counter == max_reset_attempts:
            print("警告: 多次重置後仍然有無效狀態，繼續執行但可能影響訓練")
        
        total_reward = 0
        losses = []
        done = False
        
        # 重置每個episode的統計
        env.reset_episode_stats()
        agent.reset_episode_stats()
        
        for step in range(max_steps_per_episode):
            # 選擇動作
            action_game, action_id = agent.act(state)
                    
            # 執行動作並獲取下一個狀態、獎勵等訊息
            next_state, reward, done, info = env.step(action_game)
            
            # 儲存經驗
            agent.remember(state, action_id, reward, next_state, done)
            
            
            # 訓練網絡
            loss = agent.train()
            if loss > 0:
                losses.append(loss)
            
            # 更新當前狀態和總獎勵
            state = next_state
            total_reward += reward
            # 如果回合結束，則跳出循環
            if done:
                break
                
        # 忽略過短的回合（可能是環境初始化錯誤）
        if step < 30:
            print(f"回合 {episode} 過短 (只有 {step} 步)，可能是環境錯誤 - 忽略統計")
            continue
        
        # 計算本回合用時
        episode_time = time.time() - episode_start_time
        
        # 更新epsilon
        agent.decay_epsilon()
        
        # 計算平均損失
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        # 獲取本回合的統計數據
        env_stats = env.get_episode_stats()
        agent_stats = agent.get_episode_stats()
        
        # 記錄本回合是否撞車或完成賽道
        episode_crashed = info.get('crashed', False)
        episode_completed = info.get('lap_count', 0) > 0
        
        # 檢查是否完成了一圈，如果完成則保存模型快照並發送通知
        if episode_completed:
            # 保存完成一圈時的模型和經驗回放快照
            lap_model_path = os.path.join(model_dir, f"lap_completed_model_ep{episode}.pth")
            agent.save(lap_model_path)
            print(f"已保存完成一圈的模型快照: {lap_model_path}")
            
            if email_config['enabled'] and email_config['notify_lap_completion']:
                lap_completion_count += 1
            # 準備圖表作為附件
            plot_path = os.path.join(model_dir, f"lap_completion_{lap_completion_count}.png")
            plot_training_progress(episode, total_rewards, avg_rewards, mean_q_values, 
                                    crash_counts, completed_laps, mean_speeds, 
                                    speed_distribution, completion_percentages, save_dir=model_dir)
            
            # 準備通知內容
            subject = f"賽車訓練通知: 完成一圈! (第{lap_completion_count}次)"
            message = (f"模型在訓練過程中成功完成了一圈!\n\n"
                      f"完成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                      f"訓練回合: {episode}/{num_episodes}\n"
                      f"單圈時間: {info['lap_time']:.2f}秒\n"
                      f"單圈距離: {info['travel_distance']:.2f}\n"
                      f"總獎勵: {total_reward:.2f}\n"
                      f"平均速度: {env_stats['speed_stats']['mean_speed']:.2f}\n\n"
                      f"這是模型第{lap_completion_count}次完成一圈。")
            
            # 發送通知
            send_email_notification(
                subject=subject,
                message=message,
                attachments=[("training_progress.png", plot_path)]
            )
        
        # 更新監控指標
        total_rewards.append(total_reward)
        
        # 計算移動平均獎勵
        window = min(100, len(total_rewards))
        avg_reward = np.mean(total_rewards[-window:])
        avg_rewards.append(avg_reward)
        
        # 保存統計數據
        mean_q_values.append(agent_stats['avg_q'])
        crash_counts.append(1 if episode_crashed else 0)
        completed_laps.append(1 if episode_completed else 0)
        mean_speeds.append(env_stats['speed_stats']['mean_speed'])
        
        # 計算並記錄完成百分比
        track_samples_size = 1963  
        completion_percentage = min(100, (info['travel_distance'] / track_samples_size) * 100)
        completion_percentages.append(completion_percentage)
        
        # 根據完成率調整epsilon (如果完成率>70%，將epsilon設為0.01)
        if completion_percentage > 70 and agent.epsilon > 0.01:
            old_epsilon = agent.epsilon
            agent.epsilon = 0.01
            print(f"因完成率達{completion_percentage:.1f}%，將epsilon從{old_epsilon:.4f}降至0.01")
        
        # 更新速度分布統計
        if speed_dist_counts == 0:
            speed_distribution = env_stats['speed_distribution']
        else:
            speed_distribution = (speed_distribution * speed_dist_counts + env_stats['speed_distribution']) / (speed_dist_counts + 1)
        speed_dist_counts += 1
        
        # 定期保存模型
        if episode % save_freq == 0 or episode == num_episodes:
            save_path = os.path.join(model_dir, f"agent_ep{episode}.pth")
            agent.save(save_path)
            
            # 繪製訓練進度圖
            plot_training_progress(episode, total_rewards, avg_rewards, mean_q_values, 
                                      crash_counts, completed_laps, mean_speeds, 
                                      speed_distribution, completion_percentages, save_dir=model_dir)
            
        
        # 打印訓練訊息
        if episode % log_interval == 0:
            # 格式化輸出訓練訊息
            print(f"Episode {episode}/{num_episodes} - "
                  f"Steps: {step+1}, "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}, "
                  f"Loss: {avg_loss:.6f}\n"
                  f"Mean Q-value: {agent_stats['avg_q']:.2f}, "
                  f"Mean Speed: {env_stats['speed_stats']['mean_speed']:.2f}, "
                  f"Crash Count (total): {env.crash_count}, "
                  f"Travel Distance: {info['travel_distance']:.2f}, "
                  f"Time: {episode_time:.2f}s")
            
            
            # 如果此回合完成了一圈
            if episode_completed:
                print(f"成功完成賽道! 單圈時間: {info['lap_time']:.2f}s, 單圈距離: {info['travel_distance']:.2f}")
            else:
                # 如果沒有完成，顯示完成百分比
                if 'travel_distance' in info:
                    # 使用checkpoint索引計算完成百分比
                    track_samples_size = 1963  # 軌道樣本總數
                    completion_percentage = min(100, (info['travel_distance'] / track_samples_size) * 100)
                    print(f"賽道完成度: {completion_percentage:.1f}%")
        
        
        # 記錄完成百分比
        if not episode_completed and 'travel_distance' in info:
            track_samples_size = 1963  # 與上面保持一致
            completion_percentage = min(100, (info['travel_distance'] / track_samples_size) * 100)
        
        # 記錄到CSV文件
        csv_data = {
            'episode': episode,
            'steps': step+1,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'epsilon': agent.epsilon,
            'loss': avg_loss,
            'mean_q': agent_stats['avg_q'],
            'mean_speed': env_stats['speed_stats']['mean_speed'],
            'crash_count': env.crash_count,
            'completed': 1 if episode_completed else 0
        }  
        with open(csv_log_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer_csv = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer_csv.writerow(csv_data)
    
    # 總訓練時間
    total_training_time = time.time() - start_training_time
    print(f"總訓練時間: {total_training_time:.2f}s, 平均每個回合: {total_training_time/num_episodes:.2f}s")
    
    # 繪製最終訓練進度圖
    plot_training_progress(num_episodes, total_rewards, avg_rewards, mean_q_values, 
                              crash_counts, completed_laps, mean_speeds, 
                              speed_distribution, completion_percentages, save_dir=model_dir, final=True)

    # 記錄訓練摘要
    summary_path = os.path.join(log_dir, f"training_summary_{agent.__class__.__name__}_{time.strftime('%Y%m%d-%H%M%S')}.txt")
    with open(summary_path, 'w', encoding='utf-8-sig') as f:
        f.write(f"訓練摘要 - {agent.__class__.__name__}\n")
        f.write(f"訓練時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"總回合數: {num_episodes}\n")
        f.write(f"總訓練時間: {total_training_time:.2f}秒\n")
        f.write(f"平均每回合時間: {total_training_time/num_episodes:.2f}秒\n")
        f.write(f"最終獎勵: {total_rewards[-1]:.2f}\n")
        f.write(f"最終平均獎勵: {avg_rewards[-1]:.2f}\n")
        f.write(f"總撞車次數: {sum(crash_counts)}\n")
        f.write(f"總完成圈數: {sum(completed_laps)}\n")
        f.write(f"完成率: {sum(completed_laps)/num_episodes*100:.2f}%\n")
        f.write(f"平均速度: {np.mean(mean_speeds):.2f}\n")
        f.write(f"最終Q值: {mean_q_values[-1]:.2f}\n")
        f.write("\n速度分布:\n")
        for i, speed_bin in enumerate(['0-30', '30-60', '60-90', '90+']):
            f.write(f"  {speed_bin}: {speed_distribution[i]*100:.2f}%\n")
    print(f"訓練摘要已保存至: {summary_path}")
    
    # 發送訓練完成通知
    if email_config['enabled'] and email_config['notify_training_completion']:
        subject = "賽車訓練通知: 訓練已完成"
        message = (f"賽車訓練已完成!\n\n"
                  f"完成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"總回合數: {num_episodes}\n"
                  f"總訓練時間: {total_training_time:.2f}秒\n"
                  f"平均每回合時間: {total_training_time/num_episodes:.2f}秒\n"
                  f"最終獎勵: {total_rewards[-1]:.2f}\n")
        
        # 發送帶附件的郵件
        # After the final plot_training_progress call
        final_plot_path = os.path.join(model_dir, "final_training_stats.png")
        send_email_notification(
            subject=subject,
            message=message,
            attachments=[
                ("final_training_stats.png", final_plot_path),
                ("training_summary.txt", summary_path)
            ]
        )
    
    return total_rewards, avg_rewards

def plot_rewards(rewards, avg_rewards=None, window=100):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    
    if avg_rewards is None and len(rewards) >= window:
        # 計算移動平均
        avg_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        plt.plot(avg_rewards, label=f'Moving Average ({window} ep)', color='red')
    elif avg_rewards is not None:
        plt.plot(range(len(avg_rewards)), avg_rewards, label='Average Reward', color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_progress.png")
    plt.show()

def plot_training_progress(episode, rewards, avg_rewards, q_values, crashes, completions, 
                          speeds, speed_dist, completion_percentages=None, save_dir="models", final=False):
    """繪製完整的訓練進度圖表，包含所有監控指標"""
    plt.style.use('ggplot')
    
    # 確保數據是列表而非tensor
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.tolist()
    if isinstance(avg_rewards, torch.Tensor):
        avg_rewards = avg_rewards.tolist()
    if isinstance(q_values, torch.Tensor):
        q_values = q_values.tolist()
    if isinstance(crashes, torch.Tensor):
        crashes = crashes.tolist()
    if isinstance(completions, torch.Tensor):
        completions = completions.tolist()
    if isinstance(speeds, torch.Tensor):
        speeds = speeds.tolist()
    if isinstance(completion_percentages, torch.Tensor):
        completion_percentages = completion_percentages.tolist()
    
    # 檢查速度值是否為有效數值，更嚴格的檢查
    valid_speeds = False
    if speeds and len(speeds) > 0:
        # 移除非數值元素和NaN值
        filtered_speeds = []
        for s in speeds:
            if isinstance(s, numbers.Real) and not (np.isnan(s) or np.isinf(s)):
                filtered_speeds.append(float(s))
        
        # 更新speeds為過濾後的數據
        speeds = filtered_speeds
        valid_speeds = len(speeds) > 0
        
        # 確保數值合理（去除異常大或小的值）
        if valid_speeds:
            # 檢查速度範圍，剔除不合理的速度值
            speeds = [s for s in speeds if 0 <= s <= 150]  # 假設速度範圍在0-150之間
            valid_speeds = len(speeds) > 0
    

    

            

    fig = plt.figure(figsize=(20, 20))  # 增加高度
    total_rows = 4  # 4行子圖 (增加到4行來容納完成百分比)
    
    # 1. 獎勵曲線
    ax1 = fig.add_subplot(total_rows, 2, 1)
    if rewards and len(rewards) > 0:
        ax1.plot(rewards, 'b-', alpha=0.5, label='Episode Reward')
    if avg_rewards and len(avg_rewards) > 0:
        ax1.plot(avg_rewards, 'r-', label='Avg Reward (100 ep)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q值變化
    ax2 = fig.add_subplot(total_rows, 2, 2)
    if q_values and len(q_values) > 0 and all(isinstance(x, (int, float)) for x in q_values if x is not None):
        # 過濾None和NaN值
        valid_q_values = [x for x in q_values if x is not None and not (isinstance(x, float) and np.isnan(x))]
        if valid_q_values:
            ax2.plot(valid_q_values, 'g-', label='Mean Q-Value')
            # 計算Q值的移動平均
            window = min(100, len(valid_q_values))
            if window > 0:
                q_moving_avg = [np.mean(valid_q_values[max(0, i-window):i+1]) for i in range(len(valid_q_values))]
                ax2.plot(q_moving_avg, 'r-', label='Moving Avg')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Q-Value')
            ax2.set_title('Mean Q-Values')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No valid Q-value data', ha='center', va='center')
    else:
        ax2.text(0.5, 0.5, 'No valid Q-value data', ha='center', va='center')
    
    # 3. 撞車次數和完成率
    ax3 = fig.add_subplot(total_rows, 2, 3)
    
    if crashes and len(crashes) > 0:
        # 計算累積撞車次數
        cum_crashes = np.cumsum(crashes)
        crash_rate = np.array(crashes)
        
        # 計算移動平均撞車率（每100回合）
        window = min(50, len(crash_rate))
        if window > 0:
            crash_moving_avg = [np.mean(crash_rate[max(0, i-window):i+1]) for i in range(len(crash_rate))]
            ax3.plot(crash_moving_avg, 'r-', label='Crash Rate (Moving Avg)')
    
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Crash Rate')
        ax3.set_title('Crash Statistics')
        ax3.legend()
        
        # 第二個y軸顯示累積撞車數
        ax3b = ax3.twinx()
        ax3b.plot(cum_crashes, 'b--', label='Cumulative Crashes')
        ax3b.set_ylabel('Total Crashes')
        ax3b.legend(loc='upper right')
    else:
        ax3.text(0.5, 0.5, 'No crash data available', ha='center', va='center')
    
    # 4. 完成率
    ax4 = fig.add_subplot(total_rows, 2, 4)
    
    if completions and len(completions) > 0:
        # 計算累積完成數
        cum_completions = np.cumsum(completions)
        completion_rate = np.array(completions)
        
        # 計算移動平均完成率
        window = min(50, len(completion_rate))
        if window > 0:
            completion_moving_avg = [np.mean(completion_rate[max(0, i-window):i+1]) for i in range(len(completion_rate))]
            ax4.plot(completion_moving_avg, 'g-', label='Completion Rate (Moving Avg)')
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Completion Rate')
        ax4.set_title('Lap Completion Statistics')
        ax4.legend()
        
        # 第二個y軸顯示累積完成數
        ax4b = ax4.twinx()
        ax4b.plot(cum_completions, 'b--', label='Cumulative Completions')
        ax4b.set_ylabel('Total Completions')
        ax4b.legend(loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'No completion data available', ha='center', va='center')
    
    # 5. 平均速度
    ax5 = fig.add_subplot(total_rows, 2, 5)
    if valid_speeds:
        speed_kmh = speeds
        ax5.plot(speed_kmh, 'b-', alpha=0.5, label='Mean Speed (km/h)')
        
        # 計算速度的移動平均
        window = min(50, len(speed_kmh))
        if window > 0:
            speed_moving_avg = [np.mean(speed_kmh[max(0, i-window):i+1]) for i in range(len(speed_kmh))]
            ax5.plot(speed_moving_avg, 'r-', label='Moving Avg')
        
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Speed (km/h)')
        ax5.set_title('Mean Speed per Episode')
        ax5.legend()
    else:
        print("警告: 沒有有效的速度數據用於繪圖")
        ax5.text(0.5, 0.5, 'No valid speed data', ha='center', va='center')
    
    # 6. 速度分布
    ax6 = fig.add_subplot(total_rows, 2, 6)
    speed_bins = ['0-30', '30-60', '60-90', '90+']
    
    if isinstance(speed_dist, np.ndarray) and len(speed_dist) == 4 and not np.isnan(speed_dist).any():
        ax6.bar(speed_bins, speed_dist, color='skyblue')
        ax6.set_xlabel('Speed Range (km/h)')
        ax6.set_ylabel('Percentage')
        ax6.set_title('Speed Distribution')
        
        # 顯示百分比
        for i, v in enumerate(speed_dist):
            ax6.text(i, v + 0.01, f'{v:.1%}', ha='center')
    else:
        ax6.text(0.5, 0.5, 'No valid speed distribution data', ha='center', va='center')
    
    # 7. 賽道完成百分比
    ax7 = fig.add_subplot(total_rows, 2, 7)
    
    if completion_percentages and len(completion_percentages) > 0:
        ax7.plot(completion_percentages, 'g-', alpha=0.6, label='Completion (%)')
        
        # 計算完成百分比的移動平均
        window = min(50, len(completion_percentages))
        if window > 0:
            completion_moving_avg = [np.mean(completion_percentages[max(0, i-window):i+1]) for i in range(len(completion_percentages))]
            ax7.plot(completion_moving_avg, 'r-', label='Moving Avg')
    else:
        ax7.text(0.5, 0.5, 'No completion percentage data available', ha='center', va='center')
    
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Completion (%)')
    ax7.set_title('Track Completion Percentage')
    ax7.set_ylim([0, 105])  # 設置y軸限制為0-105%
    ax7.legend()
    plt.tight_layout()
    
    # 根據是否為最終統計決定保存文件名
    if final:
        plt.savefig(f"{save_dir}/final_training_stats.png")
    else:
        plt.savefig(f"{save_dir}/training_stats_ep{episode}.png")
    
    plt.close()  # 關閉圖片，避免memory泄漏



def main():
    global start_time
    start_time = time.time()
    
    # 注冊訊號處理器，捕獲 CTRL+C 和終止訊號
    signal.signal(signal.SIGINT, signal_handler)  # CTRL+C
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)  # 終止訊號
        
    # CUDA設置 - 避免TF32導致的inf/NaN溢出
    if torch.cuda.is_available():
        print("設置CUDA選項以避免數值溢出")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    
    parser = argparse.ArgumentParser(description='自動駕駛訓練')
    parser.add_argument('--episodes', type=int, default=4000, help='訓練回合數')
    parser.add_argument('--max_steps', type=int, default=5000, help='每個回合的最大步數 (配合timeout)') # 150s timeout / 0.04s/step = 3750 steps. 5000 allows for longer if timeout is increased.
    parser.add_argument('--batch_size', type=int, default=256, help='訓練批次大小')
    parser.add_argument('--lr', type=float, default=3e-5, help='學習率 (原為1e-4, 降低以增加穩定性)')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子') 
    parser.add_argument('--epsilon', type=float, default=1.0, help='初始探索率')
    parser.add_argument('--epsilon_decay', type=float, default=0.9975, help='指數衰減模式下的衰減率 (原為0.997)') # 輕微調整
    parser.add_argument('--epsilon_min', type=float, default=0.05, help='最小探索率') # 保持或可略增至0.05
    parser.add_argument('--epsilon_decay_mode', type=str, default='exponential', 
                        choices=['exponential', 'linear'], help='探索率衰減模式: exponential或linear')
    parser.add_argument('--decay_steps', type=int, default=1000000, help='線性衰減模式下的衰減步數 (如果使用linear mode)')
    parser.add_argument('--buffer_size', type=int, default=250000, help='經驗回放緩衝區大小 (原150k-200k, 可再略增)')
    parser.add_argument('--update_target_freq', type=int, default=7500, help='目標網路更新頻率(步數) (原為5000, 減慢更新)')
    parser.add_argument('--forward_pretraining', action='store_true', default=True, help='是否進行前進動作預訓練') # default=True
    parser.add_argument('--pretraining_episodes', type=int, default=30, help='前進動作預訓練的回合數 (減少, 讓主訓練更快開始)')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber損失的delta值 (SmoothL1Loss beta)')
    parser.add_argument('--grad_clip_norm', type=float, default=5.0, help='梯度裁剪的最大範數 (原為5.0, 可略微放寬)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW優化器的權重衰減參數 (稍小)')

    parser.add_argument('--save_freq', type=int, default=100, help='模型保存頻率(回合)')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存目錄')
    parser.add_argument('--load_model', type=str, default=None, help='載入已有模型路徑')
    parser.add_argument('--exe_path', type=str, default='racing.exe', help='賽車遊戲可執行文件路徑')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Replay-buffer 預熱步數') 
    
    # 添加電子郵件通知相關選項
    parser.add_argument('--email_notifications', action='store_true', help='啟用電子郵件通知')
    parser.add_argument('--email_recipient', type=str, default='C110110157@gmail.com', help='電子郵件收件人')
    parser.add_argument('--email_sender', type=str, default='training.notification@gmail.com', help='發件人電子郵件')
    parser.add_argument('--email_password', type=str, default='', help='發件人電子郵件密碼或應用密碼')
    parser.add_argument('--notify_lap_completion', action='store_true', help='完成一圈時發送通知')
    parser.add_argument('--notify_training_completion', action='store_true', help='訓練完成時發送通知')
    parser.add_argument('--notify_errors', action='store_true', help='發生錯誤時發送通知')
    parser.add_argument('--test_email', action='store_true', help='發送測試郵件並退出程式')
    
    args = parser.parse_args()
    
    # 初始化電子郵件通知配置
    global email_config
    email_config = {
        'enabled': args.email_notifications,
        'recipient': args.email_recipient,
        'sender': args.email_sender,
        'password': args.email_password,
        'notify_lap_completion': args.notify_lap_completion,
        'notify_training_completion': args.notify_training_completion,
        'notify_errors': args.notify_errors
    }
    
    # 如果指定了測試郵件選項，發送測試郵件然後退出
    if args.test_email:
        print("正在發送測試郵件...")
        if not args.email_password:
            print("錯誤: 使用--email_password參數提供Gmail應用程式密碼")
            return
            
        success = send_email_notification(
            subject="賽車訓練通知: 測試郵件",
            message="這是一封測試郵件，用於確認電子郵件通知功能正常運作。\n\n"
                   f"發送時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"發件人: {args.email_sender}\n"
                   f"收件人: {args.email_recipient}"
        )
        
        if success:
            print("測試郵件發送成功! 請檢查您的收件箱。")
        else:
            print("測試郵件發送失敗。請檢查上方錯誤訊息。")
            
        # 測試郵件記錄到logs.csv
        log_to_csv(
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            runtime=time.time() - start_time,
            reason="測試郵件發送完成"
        )
        return
    
    # 環境初始化
    env = RacingEnvironment(exe_path=args.exe_path)
    state_size = 9
    action_size = env.action_size
    
    
    # 根據選擇初始化對應的智能體
    agent = DualOutputDQNAgent(
            state_size=state_size,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon, # 使用 args.epsilon (預設為 1.0)
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            epsilon_decay_mode=args.epsilon_decay_mode,
            decay_steps=args.decay_steps,
            huber_delta=args.huber_delta,
            grad_clip_norm=args.grad_clip_norm,
            weight_decay=args.weight_decay,
            update_target_freq=args.update_target_freq
        )
    
    # 如果指定了模型路徑，則載入模型
    if args.load_model:
        agent.load(args.load_model)
    
    try:
        # 開始訓練
            rewards, avg_rewards = train_agent(
                agent=agent,
                env=env,
                num_episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
                save_freq=args.save_freq,
                model_dir=args.model_dir,
                warmup_steps=args.warmup_steps,
                forward_pretraining=args.forward_pretraining,
                pretraining_episodes=args.pretraining_episodes,
                turn_training_episodes=0  # 設置為0，不使用轉向訓練
            )
            
            # 繪製訓練獎勵圖
            plot_rewards(rewards, avg_rewards)
            
            # 記錄訓練完成
            log_to_csv(
                time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                runtime=time.time() - start_time,
                reason=f"訓練順利完成 (共{args.episodes}回合)"
            )
        
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"執行過程中出錯: {error_msg}")
        print(traceback_str)
        
        # 記錄錯誤到logs.csv
        log_to_csv(
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            runtime=time.time() - start_time,
            reason=f"訓練錯誤中斷: {error_msg}"
        )
        
        # 發送錯誤通知
        if 'email_config' in globals() and email_config['enabled'] and email_config['notify_errors']:
            subject = "賽車訓練通知: 訓練出現錯誤"
            message = (f"訓練程式出現錯誤，已自動停止!\n\n"
                      f"錯誤時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                      f"錯誤訊息: {error_msg}\n\n"
                      f"詳細堆棧跟踪:\n{traceback_str}")
            
            # 嘗試獲取當前圖表作為附件
            attachments = []
            try:
                if 'model_dir' in locals() and os.path.exists(args.model_dir):
                    plot_files = [f for f in os.listdir(args.model_dir) if f.endswith('.png')]
                    if plot_files:
                        latest_plot = sorted(plot_files)[-1]  # 獲取最新的圖表
                        plot_path = os.path.join(args.model_dir, latest_plot)
                        attachments.append(("error_training_state.png", plot_path))
            except Exception:
                # 如果獲取附件出錯，忽略並繼續發送郵件
                pass
                
            send_email_notification(
                subject=subject,
                message=message,
                attachments=attachments
            )
    finally:
        # 關閉環境
        env.close()

if __name__ == "__main__":
    main()
