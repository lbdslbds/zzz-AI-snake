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
import pytesseract
import signal
import keyboard
from datetime import datetime
import sys

# 全局变量
running = Value(c_bool, True)
training_enabled = Value(c_bool, True)

def signal_handler(signum, frame):
    global running
    print("Interrupt received, stopping...")
    running.value = False

# 设置信号处理器
signal.signal(signal.SIGINT, signal_handler)
training_enabled = Value('b', True)
class GameAI(nn.Module):
    def __init__(self):
        super(GameAI, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 outputs for W, A, S, D
        self.fc3 = nn.Linear(128, 1)  # 1 output for state

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        action = self.fc2(x)
        state = torch.sigmoid(self.fc3(x))
        return action, state
    
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
            model = GameAI()  # 创建新的模型实例
            optimizer = optim.Adam(model.parameters())  # 创建新的优化器
    else:
        print(f"No checkpoint found at {path}. Creating a new model.")
        model = GameAI()  # 创建新的模型实例
        optimizer = optim.Adam(model.parameters())  # 创建新的优化器
    
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
    image = screen[60:170, 1020:1890]
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化 (使用 THRESH_BINARY 而不是 THRESH_BINARY_INV)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # 创建保存图像的目录
    save_dir = 'C:/Users/Administrator/Desktop/zzzai/score_images'
    roi_save_dir = 'C:/Users/Administrator/Desktop/zzzai/roi_images'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(roi_save_dir, exist_ok=True)
    
    # 每5秒保存一次图像
    current_time = time.time()
    if not hasattr(get_score, 'last_save_time') or current_time - get_score.last_save_time >= 5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'score_image_{timestamp}.png'
        cv2.imwrite(os.path.join(save_dir, filename), binary)
        get_score.last_save_time = current_time
        print(f"Score image saved: {filename}")
    
    # 加载数字模板
    templates = load_templates()
    
    # 获取模板的大小
    template_height, template_width = next(iter(templates.values())).shape
    
    # 查找每个数字
    digits = []
    for x in range(0, binary.shape[1] - template_width + 1, 58):  # 步长设为59，即半个数字宽度
        roi = binary[:, x:x+template_width]
        
        # 确保ROI至少与模板一样大
        if roi.shape[0] < template_height:
            roi = cv2.copyMakeBorder(roi, 0, template_height - roi.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
        
        # 保存ROI图像
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # roi_filename = f'roi_{timestamp}_{x}.png'
        # cv2.imwrite(os.path.join(roi_save_dir, roi_filename), roi)
        #print(f"ROI image saved: {roi_filename}")
        
        best_match = -1
        best_val = -np.inf
        for digit, template in templates.items():
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_match = digit
        if best_val > 0.4:  # 降低阈值以增加检测的灵敏度
            digits.append(str(best_match))
    
    score = ''.join(digits)
    # print(f"Extracted score: {score}")
    
    try:
        return int(score)
    except ValueError:
        return 0

def process_screen(screen):
    # Crop and resize the screen
    screen = screen[251:1385, 630:1927]
    screen = cv2.resize(screen, (84, 84))
    
    # Save the processed screen image
    save_dir = 'C:/Users/Administrator/Desktop/zzzai/processed_screens'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save every 5 seconds
    current_time = time.time()
    if not hasattr(process_screen, 'last_save_time') or current_time - process_screen.last_save_time >= 5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'processed_screen_{timestamp}.png'
        cv2.imwrite(os.path.join(save_dir, filename), screen)
        process_screen.last_save_time = current_time
        print(f"Processed screen image saved: {filename}")
    
    # Convert to PyTorch tensor
    screen = np.transpose(screen, (2, 0, 1))
    processed = torch.FloatTensor(screen).unsqueeze(0)
    print(f"Processed screen shape: {processed.shape}")
    
    return processed

def screen_capture_process(queue, stop_event):
    while not stop_event.is_set():
        screen = get_screen()
        score = get_score(screen)
        queue.put((screen, score))
        time.sleep(0.1)  # Adjust this value to control capture frequency

def detect_death(screen, template):
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val > 0.8  # Adjust this threshold as needed

def death_detection_thread(screen_queue, is_dead, stop_event):
    death_template = cv2.imread('C:/Users/Administrator/Desktop/zzzai/death_image.png', 0)  # Load your death image here
    while not stop_event.is_set():
        if not screen_queue.empty():
            screen, _ = screen_queue.get()
            gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            if detect_death(gray_screen, death_template):
                is_dead.value = True
        time.sleep(0.1)  # Adjust this value to control detection frequency

def detect_start_image():
    # 加载开始图片
    start_image = cv2.imread('C:/Users/Administrator/Desktop/zzzai/start_image.png', 0)
    
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
        
        # 如果匹配度超过阈值，则认为检测到了开始图片
        if max_val > 0.8:  # 你可能需要调整这个阈值
            print("Start image detected! Starting the program...")
            return True
        
        time.sleep(0.5)  # 每0.5秒检查一次
#训练结束判定
def detect_end_game(screen_queue, running, training_enabled, stop_event):
    end_game_image = cv2.imread('C:/Users/Administrator/Desktop/zzzai/end_game_image.png', 0)
    
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
    
    while not stop_event.is_set() and running.value:
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
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    #加载数字模板
    templates = load_templates()
    MODEL_PATH = 'C:/Users/Administrator/Desktop/zzzai/game_ai_model.pth'
    print("Loading model...")
    model, optimizer = load_model(model, optimizer, MODEL_PATH)
    print("Model loaded or created successfully")

    screen_queue = Queue()
    stop_event = Event()
    is_dead = Value(c_bool, False)

    print("Starting screen capture process...")
    screen_process = Process(target=screen_capture_process, args=(screen_queue, stop_event))
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
    previous_score = 0

    # def restart_game():
    #     nonlocal update_model, is_dead
    #     print("Restarting game...")
    #     time.sleep(2)
    #     keyboard.press('j')
    #     print('j')
    #     time.sleep(0.1)
    #     keyboard.release('j')
    #     time.sleep(1)  # Wait 4 seconds
    #     update_model = True
    #     is_dead.value = False
    #     print("Game restarted")
    def restart_game():
        nonlocal update_model, is_dead
        print("Restarting game...")
        time.sleep(2)
        keyboard.press('j')
        print('j')
        time.sleep(0.1)
        keyboard.release('j')
        
        # Load the specific image to look for
        restart_image = cv2.imread('C:/Users/Administrator/Desktop/zzzai/start_image.png', 0)
        
        if restart_image is None:
            print("Error: Unable to load restart image. Please check the file path.")
            return
        
        print("Waiting for restart image...")
        max_wait_time = 10  # Maximum wait time in seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Capture the screen
            screenshot = pyg.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(screenshot, restart_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # If the match is good enough, consider it found
            if max_val > 0.8:  # Adjust this threshold as needed
                print("Restart image detected!")
                update_model = True
                is_dead.value = False
                print("Game restarted")
                return
            else:
                keyboard.press('j')
                print('j')
                time.sleep(0.1)
                keyboard.release('j')
            time.sleep(0.5)  # Wait for 0.5 seconds before checking again
        
        print("Restart image not detected within the time limit.")
        # You might want to handle this case, maybe by retrying or alerting the user
        print("Entering main loop")
    try:
        while running.value and not stop_event.is_set():
            if is_dead.value:
                print(f"Death detected. Step: {step}")
                step += 1
                restart_game()
                continue
        
            if screen_queue.empty():
                time.sleep(0.1)
                continue

            screen, current_score = screen_queue.get()
            processed_screen = process_screen(screen)
            action_logits, state = model(processed_screen)
            
            # 使用 softmax 来获取动作概率
            action_probs = torch.softmax(action_logits, dim=1)
            
            # 选择概率最高的动作
            action = torch.argmax(action_probs, dim=1).item()
            actions = ['W', 'A', 'S', 'D']
            chosen_action = actions[action]
            
            if not is_dead.value:
                keyboard.press(chosen_action.lower())
                time.sleep(0.01)
                keyboard.release(chosen_action.lower())
                
                if state.item() > 0.5:
                    keyboard.press('j')
                else:
                    keyboard.release('j')
            
            print(f"Step: {step}, Score: {current_score}, Action: {chosen_action}, State: {state.item():.2f}, Training: {'Enabled' if training_enabled.value else 'Disabled'}")
            
            if update_model and training_enabled.value:
                reward = current_score - previous_score
                target_q = reward + 0.99 * torch.max(action_logits).detach()
                current_q = action_logits[0, action]
                
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print("Training is currently disabled")
            previous_score = current_score
            
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

        # Stop screen process
        screen_process.join(timeout=5)
        if screen_process.is_alive():
            print("Screen process did not stop in time, terminating...")
            screen_process.terminate()
        print("Screen process stopped")

        # Stop death thread (set as daemon)
        death_thread.join(timeout=5)
        if death_thread.is_alive():
            print("Death thread is still running, but it's a daemon thread")
        print("Death thread stopped")

        # Stop end game thread (set as daemon)
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
