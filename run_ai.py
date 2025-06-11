import torch
import numpy as np
import time
import os
import sys
from game_interface import GameInterface
from train import DQN, DQNAgent

def main():
    print("=" * 60)
    print("賽車 AI 控制器")
    print("=" * 60)
    print("重要說明:")
    print("1. 請先確保已經通過 train.py 訓練出模型")
    print("2. 目前有兩種使用方式:")
    print("   * 方法A - 命令行模式: racing.exe MODE_AI (AI自動連接)")
    print("   * 方法B - 界面模式: 先啟動此程式，再啟動racing.exe，選擇1vPC")
    print("=" * 60)
    
    # Path to your trained model
    model_path = "models/dqn_model.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"錯誤: 模型在 {model_path} 未找到")
        print(f"請先執行 train.py 訓練模型")
        return
        
    print(f"載入模型從 {model_path}")
    
    # Create agent
    agent = DQNAgent()
    
    # Load trained model
    if not agent.load(model_path):
        print("載入模型失敗")
        return
        
    print("模型載入成功")
    agent.epsilon = 0.0  # Disable exploration for inference
    
    # Create game interface
    interface = GameInterface()
    
    print("\n請選擇你想要的流程:")
    print("1. 我將先啟動遊戲 racing.exe MODE_AI，再與此程式連接")
    print("2. 我想先啟動此程式，再啟動遊戲並選擇1vPC選項")
    
    # 獲取使用者輸入
    mode_choice = "0"
    while mode_choice not in ["1", "2"]:
        mode_choice = input("請選擇 (1/2): ").strip()
    
    if mode_choice == "1":
        # 使用方法A: 命令行模式
        print("\n您選擇了命令行模式。")
        print("請確保已使用命令 racing.exe MODE_AI 啟動遊戲。")
        
        attempt_count = 0
        max_attempts = 10
        connected = False
        
        for attempt in range(1, max_attempts + 1):
            print(f"連接嘗試 {attempt}/{max_attempts}...")
            if interface.connect(max_attempts=1, retry_interval=1.0):
                connected = True
                print("成功連接到遊戲!")
                break
            time.sleep(1)
            
        if not connected:
            print("無法連接到遊戲。請確保遊戲已經以 MODE_AI 模式啟動。")
            return
    else:
        # 使用方法B: 界面模式
        print("\n您選擇了界面模式。接下來請按照這些步驟操作:")
        print("1. 保持此程式運行")
        print("2. 啟動遊戲 racing.exe (不要使用MODE_AI參數)")
        print("3. 在主選單選擇 1vPC 選項")
        print("4. 程式將嘗試建立連接")
        print("\n遊戲可能需要修改才能支援這種模式。如果無法連接，請參考故障排除建議。")
        
        input("按 Enter 鍵開始等待遊戲連接...")
        
        print("\n等待遊戲啟動並選擇1vPC模式中...")
        print("(如果長時間無法連接，可能需要修改遊戲程式)")
        
        # 持續嘗試連接直到成功
        connected = False
        attempt_count = 0
        
        host = interface.host
        port = interface.port
        print(f"等待遊戲連接到 {host}:{port}")
        print("請在遊戲中選擇 1vPC 選項")
        
        while not connected:
            try:
                attempt_count += 1
                sys.stdout.write(f"\r嘗試連接 ({attempt_count})...")
                sys.stdout.flush()
                
                connected = interface.connect(max_attempts=1, retry_interval=0.5)
                
                if not connected:
                    time.sleep(2)  # 每2秒嘗試一次
                    
                    # 每15次嘗試顯示一次故障排除建議
                    if attempt_count % 15 == 0:
                        print("\n\n長時間無法連接，可能原因:")
                        print("1. 遊戲程式未正確實現1vPC模式的網絡連接")
                        print("2. 防火牆阻擋了連接")
                        print("3. 端口配置不匹配\n")
                        print("建議解決方案:")
                        print("* 方案一: 使用命令行模式 (racing.exe MODE_AI)")
                        print("* 方案二: 查看提供的修改建議，修改 Final_Project.cpp")
                        print("* 方案三: 嘗試不同的連接順序\n")
                        
                        continue_waiting = input("繼續等待連接? (y/n): ").strip().lower()
                        if continue_waiting != 'y':
                            print("結束等待，程式將退出")
                            return
                        
                        print("\n繼續等待遊戲連接...")
                        
            except KeyboardInterrupt:
                print("\n使用者中斷連接嘗試")
                return
            except Exception as e:
                print(f"\n連接時發生錯誤: {e}")
                print("\n可能需要修改 Final_Project.cpp:")
                print("1. 確保1vPC模式與MODE_AI模式使用相同的網絡初始化")
                print("2. 檢查1vPC模式是否正確啟動網路服務器")
                print("3. 確認連接埠號碼是否與GameInterface一致 (預設12345)")
                
                retry = input("是否繼續嘗試連接? (y/n): ").strip().lower()
                if retry != 'y':
                    print("程序結束")
                    return
        
        print("\n成功連接到遊戲!")
    
    # 嘗試偵測遊戲狀態
    print("等待遊戲開始...")
    game_started = False
    waiting_time = 0
    
    while not game_started and waiting_time < 30:  # 最多等待30秒
        try:
            # 獲取當前狀態
            state = interface.get_state()
            
            # 從速度判斷遊戲是否已經開始
            if abs(state[4]) > 0.1:  # 如果速度大於0.1，假設遊戲已開始
                game_started = True
                print("\n檢測到車輛移動! 開始AI控制")
            else:
                # 取得比賽資訊
                lap_info = interface.get_lap_info()
                
                # 如果lap_time > 0，表示比賽已經開始
                if lap_info and lap_info[1] > 0:
                    game_started = True
                    print("\n檢測到比賽開始! 開始AI控制")
                else:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    waiting_time += 0.5
            
            time.sleep(0.5)
        except Exception as e:
            print(f"\n等待遊戲開始時發生錯誤: {e}")
            print("嘗試繼續等待...")
            waiting_time += 1
            time.sleep(1)
    
    if not game_started:
        print("\n遊戲似乎尚未開始。無論如何，現在開始AI控制...")
    
    # 主控制循環
    print("\n開始AI控制...")
    try:
        last_message_time = time.time()
        lap_count = 0
        
        while True:
            try:
                # Get current state from game
                state = interface.get_state()
                
                # Get lap info
                lap_info = interface.get_lap_info()
                if lap_info and lap_info[0] > lap_count:
                    lap_count = lap_info[0]
                    print(f"\n完成第 {lap_count} 圈!")
                
                # Get action from model
                steering_action, throttle_action, game_action = agent.act(state, training=False)
                
                # Print current speed and action for monitoring (limit updates to not flood console)
                current_time = time.time()
                if current_time - last_message_time > 0.5:  # 每0.5秒更新一次顯示
                    speed_kmh = state[4] * 3.6
                    dist_center = state[7]
                    track_angle = state[8]
                    sys.stdout.write(f"\r速度: {speed_kmh:.1f} km/h | 動作: {game_action} | 距中心: {dist_center:.2f}m | 角度: {track_angle:.1f}° | 圈數: {lap_count}")
                    sys.stdout.flush()
                    last_message_time = current_time
                
                # Send action to game
                success = interface.send_action(game_action)
                if not success:
                    print("\n發送動作失敗，嘗試重新連接...")
                    interface.disconnect()
                    time.sleep(1)
                    if not interface.connect(max_attempts=5, retry_interval=1.0):
                        print("重新連接失敗。退出。")
                        break
                    print("重新連接成功，繼續控制")
                
                # Small delay to not overload the connection
                time.sleep(0.05)
                
            except Exception as e:
                print(f"\n運行時出錯: {e}")
                # 嘗試重新連接
                try:
                    interface.disconnect()
                    time.sleep(1)
                    print("嘗試重新連接...")
                    if interface.connect(max_attempts=5, retry_interval=1.0):
                        print("重新連接成功，繼續控制")
                    else:
                        print("重新連接失敗，等待下一次嘗試")
                except Exception as e2:
                    print(f"重新連接時發生錯誤: {e2}")
                time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nAI控制被使用者中斷")
    finally:
        # Disconnect from game
        try:
            interface.disconnect()
            print("\n已斷開與遊戲的連接")
        except:
            print("\n斷開連接時發生錯誤")

if __name__ == "__main__":
    main()
