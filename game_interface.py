import socket
import struct
import time
import numpy as np

class GameInterface:
    """
    遊戲介面類，處理Python與C++賽車遊戲之間的通信
    
    使用簡單的TCP socket連接實現進程間通信
    """
    
    def __init__(self, host='localhost', port=12345):
        """
        初始化遊戲介面
        
        參數:
            host (str): 主機名，默認為localhost
            port (int): 端口號，默認為12345
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self, max_attempts=10, retry_interval=1.0):
        """
        連接到遊戲服務器
        
        參數:
            max_attempts (int): 最大重試次數
            retry_interval (float): 重試間隔(秒)
            
        返回:
            bool: 連接是否成功
        """
        attempts = 0
        
        while attempts < max_attempts and not self.connected:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.connected = True
                print(f"成功連接到遊戲服務器 {self.host}:{self.port}")
                return True
            except socket.error as e:
                print(f"連接失敗(嘗試 {attempts+1}/{max_attempts}): {e}")
                attempts += 1
                if attempts < max_attempts:
                    time.sleep(retry_interval)
                    
        print("無法連接到遊戲服務器，請確保遊戲已啟動且處於AI模式")
        return False
    
    def disconnect(self):
        """關閉與遊戲的連接"""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False
            print("已斷開與遊戲服務器的連接")
    
    def get_state(self):
        """
        從遊戲獲取當前狀態
        
        返回:
            numpy.array: 包含9個元素的狀態向量
            [x, y, z, heading, speed, lateral_offset, nearest_sample_idx, 
             distance_from_center, angle_with_track]
        """
        if not self.connected:
            print("警告: 未連接到遊戲服務器")
            return np.zeros(9, dtype=np.float32)
        
        try:
            # 發送狀態請求命令
            self.socket.sendall(b'GET_STATE')
            
            # 接收回應 (9個float32值)
            data = self.socket.recv(10 * 4)  # 接收10個float32值（40字節）
            if len(data) < 10 * 4:
                print("警告: 接收到的遊戲狀態數據不完整")
                return np.zeros(9, dtype=np.float32)
            
            # 解析二進制數據為9個float值
            state = struct.unpack('ffffffffff', data)[:9]  # 只取前9個值
            return np.array(state, dtype=np.float32)
        
        except Exception as e:
            print(f"獲取遊戲狀態時發生錯誤: {e}")
            self.connected = False
            return np.zeros(9, dtype=np.float32)
    
    def send_action(self, action):
        """
        向遊戲發送動作命令
        
        參數:
            action (int): 動作ID (0-8)
                0: 無動作
                1: 加速(W)
                2: 煞車(S)
                3: 左轉(A)
                4: 右轉(D)
                5: 加速+左轉(W+A)
                6: 加速+右轉(W+D)
                7: 煞車+左轉(S+A)
                8: 煞車+右轉(S+D)
                
        返回:
            bool: 命令是否成功發送
        """
        if not self.connected:
            print("警告: 未連接到遊戲服務器")
            return False
        
        try:
            # 將動作ID轉換為二進制並發送
            self.socket.sendall(b'ACTION' + struct.pack('B', action))
            
            # 接收確認回應
            response = self.socket.recv(2)
            return response == b'OK'
        
        except Exception as e:
            print(f"發送動作時發生錯誤: {e}")
            self.connected = False
            return False
    
    def reset_game(self):
        """
        重置遊戲，重新開始賽車
        
        返回:
            bool: 重置命令是否成功發送
        """
        if not self.connected:
            print("警告: 未連接到遊戲服務器")
            return False
        
        try:
            # 發送重置命令
            self.socket.sendall(b'RESET')
            
            # 接收確認回應
            response = self.socket.recv(2)
            return response == b'OK'
        
        except Exception as e:
            print(f"重置遊戲時發生錯誤: {e}")
            self.connected = False
            return False
    
    def get_lap_info(self):
        """
        獲取賽車圈數、時間和進度資訊
        
        返回:
            tuple: (lap_count, lap_time, travel_distance, checkpoint_idx)
        """
        if not self.connected:
            print("警告: 未連接到遊戲服務器")
            return (0, 0.0, 0.0, 0)
        
        try:
            # 發送圈數資訊請求命令
            self.socket.sendall(b'LAP_INFO')
            
            # 接收回應 (1個int32 + 2個float32 + 1個int32)
            data = self.socket.recv(16)  # 4 + 4 + 4 + 4 = 16字節
            if len(data) < 16:
                print(f"警告: 接收到的圈數資訊數據不完整 ({len(data)}/16字節)")
                return (0, 0.0, 0.0, 0)
            
            # 解析二進制數據
            lap_count, lap_time, travel_distance, checkpoint_idx = struct.unpack('iffi', data)
            return (lap_count, lap_time, travel_distance, checkpoint_idx)
        
        except Exception as e:
            print(f"獲取圈數資訊時發生錯誤: {e}")
            self.connected = False
            return (0, 0.0, 0.0, 0)

# 簡單的測試代碼
if __name__ == "__main__":
    interface = GameInterface()
    
    if interface.connect():
        try:
            # 獲取初始狀態
            state = interface.get_state()
            print(f"初始狀態: {state}")
            
            # 發送一些測試動作
            for action in [1, 3, 6, 1, 1]:  # 前進, 左轉, 前進+右轉, 前進, 前進
                print(f"發送動作: {action}")
                result = interface.send_action(action)
                print(f"動作結果: {'成功' if result else '失敗'}")
                time.sleep(0.5)
                
                # 獲取新狀態
                state = interface.get_state()
                print(f"新狀態: {state}")
                
                # 獲取圈數資訊
                lap_info = interface.get_lap_info()
                print(f"圈數資訊: {lap_info}")
                
            # 重置遊戲
            print("重置遊戲")
            interface.reset_game()
            
        finally:
            interface.disconnect() 