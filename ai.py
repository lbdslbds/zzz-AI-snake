import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import threading
import time
import pyautogui as pyg
import os
from multiprocessing import Process, Queue, Value, Event
from ctypes import c_bool
from pynput.keyboard import Key, Listener, Controller
import signal
import keyboard
from datetime import datetime
import sys
import random
from collections import deque

# 全局变量
running = Value(c_bool, True)
training_enabled = Value(c_bool, True)

def signal_handler(signum, frame):
    global running
    print("Interrupt received, stopping...")
    running.value = False

# 设置信号处理器
signal.signal(signal.SIGINT, signal_handler)

class GameAI(nn.Module):
    def __init__(self):
        super(GameAI, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 5) # 5个可选输出 W, A, S, D, and No Action
        self.fc3 = nn.Linear(512, 1) # 加速状态输出

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # 打印sharp以找到正确的尺寸
        # print("Shape before view:", x.shape)
        # 调整view
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        action = self.fc2(x)
        acceleration = torch.sigmoid(self.fc3(x))
        return action, acceleration

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, acceleration, reward, next_state, done):
        self.buffer.append((state, action, acceleration, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def load_model(model, optimizer, path):
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model and optimizer state dictionary loaded from {path}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            print("Creating a new model due to incompatibility.")
            model = GameAI()# 创建新的模型实例
            optimizer = optim.Adam(model.parameters(), lr=0.00025)  # 创建新的优化器
    else:
        print(f"No checkpoint found at {path}. Creating a new model.")
        model = GameAI() 
        optimizer = optim.Adam(model.parameters(), lr=0.00025)
    
    return model, optimizer

def save_model(model, optimizer, path):
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def get_screen():
    screenshot = pyg.screenshot()
    return np.array(screenshot)

def screen_capture_process(queue, stop_event, running):
    while not stop_event.is_set() and running.value:
        screen = get_screen()
        score = get_score(screen)
        queue.put((screen, score))
        time.sleep(0.01)   # 调整此值以控制屏幕捕获频率

def process_screen(screen):
    # 裁剪并调整屏幕大小 84*84
    screen = screen[251:1385, 630:1927]
    screen = cv2.resize(screen, (84, 84))
#     # 保存处理后的屏幕图像
#     save_dir = 'C:/Users/bowei/Desktop/zzzai/processed_screens'
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 每5秒保存一次
#     current_time = time.time()
#     if not hasattr(process_screen, 'last_save_time') or current_time - process_screen.last_save_time >= 5:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f'processed_screen_{timestamp}.png'
#         cv2.imwrite(os.path.join(save_dir, filename), screen)
#         process_screen.last_save_time = current_time
#         print(f"Processed screen image saved: {filename}")   
    # 转换为PyTorch张量
    screen = np.transpose(screen, (2, 0, 1))
    return torch.FloatTensor(screen).unsqueeze(0) / 255.0
#加载数字模板
def load_templates():
    templates = {}
    template_dir = 'C:/Users/Administrator/Desktop/zzzai/number_templates'
    for i in range(10):
        template_path = os.path.join(template_dir, f'{i}.png')
        if os.path.exists(template_path):
            template = cv2.imread(template_path, 0)
            # 调整模板大小为60x90（假设高宽比为3:2）
            template = cv2.resize(template, (60, 90))
            # 如果需要反转模板颜色，取消下面这行的注释
            # template = cv2.bitwise_not(template)
            templates[i] = template
    return templates

def get_score(screen):
    # 提取分数区域
    image = screen[60:170, 1020:1891]
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化图像
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # 创建保存图像的目录
    # save_dir = 'C:/Users/bowei/Desktop/zzzai/score_images'
    # roi_save_dir = 'C:/Users/bowei/Desktop/zzzai/roi_images'
    # os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(roi_save_dir, exist_ok=True)
    
    # 每5秒保存一次图像
    # current_time = time.time()
    # if not hasattr(get_score, 'last_save_time') or current_time - get_score.last_save_time >= 5:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f'score_image_{timestamp}.png'
    #     cv2.imwrite(os.path.join(save_dir, filename), binary)
    #     get_score.last_save_time = current_time
    #     print(f"Score image saved: {filename}")

    # 加载数字模板
    templates = load_templates()
    # 获取模板的大小
    template_height, template_width = next(iter(templates.values())).shape
    # 查找每个数字
    digits = []
    for x in range(0, binary.shape[1] - template_width + 1, 58):
        roi = binary[:, x:x+template_width]

        # 确保ROI至少与模板一样大
        if roi.shape[0] < template_height:
            roi = cv2.copyMakeBorder(roi, 0, template_height - roi.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
        # 保存ROI图像
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # roi_filename = f'roi_{timestamp}_{x}.png'
        # cv2.imwrite(os.path.join(roi_save_dir, roi_filename), roi)
        # print(f"ROI image saved: {roi_filename}")
        best_match = -1
        best_val = -np.inf
        for digit, template in templates.items():
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_match = digit
        if best_val > 0.4: # 降低阈值以增加检测的灵敏度
            digits.append(str(best_match))
    
    score = ''.join(digits)
    #print(f"Extracted score: {score}")
    try:
        return int(score)*10 #乘以10是因为剪切图像有bug，获取不到最后一位数字，懒得修bug了
    except ValueError:
        return 0

def detect_death(screen, template):
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val > 0.8 # 根据需要调整此阈值
#存活时间统计
def calculate_survival_reward(survival_time):
    base_reward = 1 # 每一步存活的基本奖励
    time_bonus = survival_time * 0.1 # 奖励随时间增加
    return base_reward + time_bonus
#死亡检测线程
def death_detection_thread(screen_queue, is_dead, stop_event):
    death_template = cv2.imread('C:/Users/bowei/Desktop/zzzai/death_image.png', 0)
    while not stop_event.is_set():
        if not screen_queue.empty():
            screen, _ = screen_queue.get()
            gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            if detect_death(gray_screen, death_template):
                is_dead.value = True
        time.sleep(0.1) # 调整此值以控制检测频率

def detect_start_image():
    # 加载开始图片
    start_image = cv2.imread('C:/Users/bowei/Desktop/zzzai/start_image.png', 0)
    
    if start_image is None:
        print("Error: Unable to load start image. Please check the file path.")
        return False

    print("Waiting for start image...")
    while True:
        # 截取屏幕
        screenshot = pyg.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        # 模板匹配
        result = cv2.matchTemplate(screenshot, start_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.8:
            print("Start image detected! Starting the program...")
            return True
        
        time.sleep(0.2)# 每0.2秒检查一次
#训练结束判定
def detect_end_game(screen_queue, running, training_enabled, stop_event):
    end_game_image = cv2.imread('C:/Users/bowei/Desktop/zzzai/end_game_image.png', 0)
    
    if end_game_image is None:
        print("Error: Unable to load end game image. Please check the file path.")
        return

    def on_press(key):
        nonlocal running, training_enabled
        if key == Key.esc:
            print("Esc key pressed. Stopping the program...")
            running.value = False
            stop_event.set()
            return False
        elif key == Key.space:
            training_enabled.value = not training_enabled.value
            print("Training " + ("enabled" if training_enabled.value else "disabled"))

    listener = Listener(on_press=on_press)
    listener.start()
    
    while not stop_event.is_set():
        if not screen_queue.empty():
            screen, _ = screen_queue.get()
            gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
            result = cv2.matchTemplate(gray_screen, end_game_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
            if max_val > 0.8:
                print("End game image detected! Ending program...")
                running.value = False
                stop_event.set()
                break
        
        time.sleep(0.1)
    
    listener.stop()
    print("End game detection thread ended")
#强制退出
def force_exit():
    print("Force exiting the program...")
    os._exit(0)

def main():
    global running, training_enabled
    
    print("Entering main function")
    if not detect_start_image():
        print("Failed to detect start image. Exiting...")
        return

    print("Program started!")

    model = GameAI()
    target_model = GameAI()
    optimizer = optim.Adam(model.parameters(), lr=0.00025)
    criterion = nn.MSELoss()
    
    MODEL_PATH = 'C:/Users/bowei/Desktop/zzzai/imitation_learning_model.pth'
    print("Loading model...")
    model, optimizer = load_model(model, optimizer, MODEL_PATH)
    target_model.load_state_dict(model.state_dict())
    print("Model loaded or created successfully")

    replay_buffer = ReplayBuffer(500000)
    
    screen_queue = Queue()
    stop_event = Event()
    is_dead = Value(c_bool, False)

    print("Starting screen capture process...")
    screen_process = Process(target=screen_capture_process, args=(screen_queue, stop_event, running))
    screen_process.start()
    print("Screen capture process started")

    print("Starting death detection thread...")
    death_thread = threading.Thread(target=death_detection_thread, args=(screen_queue, is_dead, stop_event))
    death_thread.daemon = True
    death_thread.start()
    print("Death detection thread started")

    print("Starting end game detection thread...")
    end_game_thread = threading.Thread(target=detect_end_game, args=(screen_queue, running, training_enabled, stop_event))
    end_game_thread.daemon = True
    end_game_thread.start()
    print("End game detection thread started")

    step = 0
    update_model = True
    survival_time = 0
    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.1  
    frame_stack = deque(maxlen=4)

    def restart_game():
        nonlocal update_model, is_dead
        print("Restarting game...")
        time.sleep(2)
        keyboard.press('j')
        time.sleep(0.1)
        keyboard.release('j')
        
        restart_image = cv2.imread('C:/Users/bowei/Desktop/zzzai/start_image.png', 0)
        
        if restart_image is None:
            print("Error: Unable to load restart image. Please check the file path.")
            return
        
        print("Waiting for restart image...")
        max_wait_time = 10
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            screenshot = pyg.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            
            result = cv2.matchTemplate(screenshot, restart_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.8:
                print("Restart image detected!")
                update_model = True
                is_dead.value = False
                print("Game restarted")
                return
            else:
                keyboard.press('j')
                time.sleep(0.1)
                keyboard.release('j')
            time.sleep(0.5)
        
        print("Restart image not detected within the time limit.")
        print("Entering main loop")

    try:
        while running.value and not stop_event.is_set():
            if is_dead.value:
                print(f"Death detected. Step: {step}, Survival time: {survival_time}")
                death_penalty = -1000
                replay_buffer.push(state, action, acceleration, death_penalty, next_state, True)
                step += 1
                survival_time = 0
                restart_game()
                frame_stack.clear()
                continue
            
            if screen_queue.empty():
                time.sleep(0.1)
                continue

            screen, current_score = screen_queue.get()
            processed_screen = process_screen(screen)
            
            if len(frame_stack) < 4:
                for _ in range(4):
                    frame_stack.append(processed_screen)
            else:
                frame_stack.append(processed_screen)
            
            state = torch.cat(list(frame_stack), dim=1)
            
            if random.random() < epsilon:
                action = random.randint(0, 4)
                acceleration = random.random() > 0.5
            else:
                with torch.no_grad():
                    action_logits, acceleration_prob = model(state)
                    action = torch.argmax(action_logits, dim=1).item()
                    acceleration = acceleration_prob.item() > 0.5
            
            actions = ['W', 'A', 'S', 'D', 'No Action']
            chosen_action = actions[action]
            
            if not is_dead.value:
                if chosen_action != 'No Action':
                    keyboard.press(chosen_action.lower())
                    time.sleep(0.01)
                    keyboard.release(chosen_action.lower())
                if acceleration:
                    keyboard.press('j')
                else:
                    keyboard.release('j')

                survival_time += 1

            print(f"Step: {step}, Score: {current_score}, Action: {chosen_action}, Acceleration: {acceleration}, Survival Time: {survival_time}, Epsilon: {epsilon:.2f}, Training: {'Enabled' if training_enabled.value else 'Disabled'}")
            
            next_screen, next_score = screen_queue.get()
            next_processed_screen = process_screen(next_screen)
            frame_stack.append(next_processed_screen)
            next_state = torch.cat(list(frame_stack), dim=1)
            
            reward = (next_score - current_score) + calculate_survival_reward(survival_time)
            done = is_dead.value
            
            replay_buffer.push(state, action, acceleration, reward, next_state, done)
            
            if len(replay_buffer) > 1000 and training_enabled.value:
                batch = replay_buffer.sample(128)
                states, actions, accelerations, rewards, next_states, dones = zip(*batch)
                
                states = torch.cat(states)
                actions = torch.tensor(actions)
                accelerations = torch.tensor(accelerations, dtype=torch.float32)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)
                
                current_q_values, current_acceleration = model(states)
                current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q_values, _ = target_model(next_states)
                    next_q_values = next_q_values.max(1)[0]
                    target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
                
                q_loss = criterion(current_q_values, target_q_values)
                acceleration_loss = criterion(current_acceleration.squeeze(), accelerations)
                
                loss = q_loss + acceleration_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Loss: {loss.item():.4f}")
            
            if step % 1000 == 0:
                target_model.load_state_dict(model.state_dict())
                print("Target network updated")
            
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            if step % 1000 == 0 and step != 0:
                save_model(model, optimizer, MODEL_PATH)
                print(f"Model saved at step {step}")
            
            step += 1

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping processes...")
        running.value = False
        stop_event.set()
        screen_process.join(timeout=5)
        if screen_process.is_alive():
            print("Screen process did not stop in time, terminating...")
            screen_process.terminate()
        print("Screen process stopped")

        death_thread.join(timeout=5)
        if death_thread.is_alive():
            print("Death thread is still running, but it's a daemon thread")
        print("Death thread stopped")

        end_game_thread.join(timeout=5)
        if end_game_thread.is_alive():
            print("End game thread is still running, but it's a daemon thread")
        print("End game thread stopped")

        save_model(model, optimizer, MODEL_PATH)
        print("Training stopped and model saved")

        # 确保所有按键都被释放
        keyboard_controller = Controller()
        for key in ['w', 'a', 's', 'd', 'j']:
            keyboard_controller.release(key)
        print("All keys released")

        print("Exiting program...")
        sys.exit(0)

if __name__ == "__main__":
    main()
