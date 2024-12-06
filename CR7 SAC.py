import gymnasium as gym
import time
import torch
import numpy as np  # NumPy kütüphanesini ekliyoruz
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt


# Callback tanımlama
class RewardCallback(BaseCallback):
    def __init__(self, log_interval=750, reset_interval=750):
        super(RewardCallback, self).__init__()
        self.episode_rewards = []
        self.log_interval = log_interval
        self.reset_interval = reset_interval
        self.step_count = 0  # Adım sayacı
        self.current_episode_reward = 0  # Mevcut episode ödülü

    def _on_step(self) -> bool:
        # Env'den ödülleri toplamak
        self.step_count += 1
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward

        # Eğer bir episode ödülü -60'tan küçükse sıfırla
        if self.current_episode_reward < -60:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            self.locals['env'].reset()

        # Adım sayısına göre log yazdırma
        if self.step_count % self.log_interval == 0:
            print(f"Adım: {self.step_count}, Son 10 Episode Toplam Reward: {sum(self.episode_rewards[-10:])}")

        # Eğer bir episode tamamlandıysa ödülü kaydet
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            self.locals['env'].reset()

        # Belirli bir adımda ortamı sıfırlama
        if self.step_count % self.reset_interval == 0:
            print("750 adım tamamlandı, ortam sıfırlanıyor.")
            self.locals['env'].reset()

        return True


# Ortam oluşturma fonksiyonu
def create_env(env_name, render_mode=False):
    return gym.make(env_name, render_mode='human' if render_mode else None)


# Algoritma eğitimi ve değerlendirme
def train_and_evaluate(model_class, env_name, timesteps, num_trials=5, render=False):
    rewards = []
    env = create_env(env_name, render_mode=render)

    # Model oluşturma
    model = model_class("MlpPolicy", env, verbose=1, buffer_size=15000, train_freq=(100, "step"), learning_starts=400, device='cuda' if torch.cuda.is_available() else 'cpu')  # Buffer boyutunu 15000'e ayarlama ve GPU desteği

    start_time = time.time()  # Eğitim süresi başlangıcı

    for trial in range(num_trials):
        model.learn(total_timesteps=timesteps)
        episode_rewards = []
        obs, info = env.reset()

        for _ in range(1000):  # Maksimum adım
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Ödül eklenmesi sırasında tipi kontrol ediliyor
            if isinstance(reward, (list, np.ndarray)):  # Ödül bir liste veya dizi mi?
                episode_rewards.extend(reward)  # Dizi ise elemanlarını ekle
            else:
                episode_rewards.append(reward)  # Skalar ise doğrudan ekle
            
            if done or truncated:
                break

        total_reward = np.sum(episode_rewards)  # Ödülleri topla
        rewards.append(total_reward)
        print(f"Trial {trial + 1}: Total Reward = {total_reward}")

    end_time = time.time()  # Eğitim süresi bitişi
    training_time = end_time - start_time  # Toplam eğitim süresi
    env.close()
    return model, rewards, training_time


# Algoritma sonuçlarını çizim fonksiyonu
def plot_algorithm_results(algorithm_name, rewards, training_time):
    avg_reward = np.mean(rewards)
    total_reward = np.sum(rewards)

    plt.figure(figsize=(12, 7))
    plt.plot(rewards, label=f"{algorithm_name} Rewards", marker='o')

    # Eğitim bilgilerini grafiğe ekleme
    plt.text(0.95, 0.85, f"Average Reward: {avg_reward:.2f}", fontsize=12, ha='right', transform=plt.gca().transAxes)
    plt.text(0.95, 0.80, f"Total Reward: {total_reward:.2f}", fontsize=12, ha='right', transform=plt.gca().transAxes)
    plt.text(0.95, 0.75, f"Training Time: {training_time:.2f} sec", fontsize=12, ha='right', transform=plt.gca().transAxes)

    plt.title(f"{algorithm_name} Performance on CarRacing-v3")
    plt.xlabel("Trials")
    plt.ylabel("Total Rewards")
    plt.legend()
    plt.grid()
    plt.show()


# Ana program
if __name__ == "__main__":
    ENV_NAME = "CarRacing-v3"  # CarRacing-v3 ortamı
    TIMESTEPS = 500  # Her trial için adım sayısı
    NUM_TRIALS = 4
    RENDER = True  # Render aktif etmek için True yapabilirsiniz

    # SAC
    print("\n--- SAC Eğitim ve Değerlendirme ---")
    sac_model, sac_rewards, sac_time = train_and_evaluate(SAC, ENV_NAME, TIMESTEPS, NUM_TRIALS, render=RENDER)
    plot_algorithm_results("SAC", sac_rewards, sac_time)
