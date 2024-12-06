import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import A2C
import matplotlib.pyplot as plt

# Ortam oluşturma fonksiyonu
def create_env(env_name, render_mode=False):
    return gym.make(env_name, render_mode='human' if render_mode else None)

# Algoritma eğitimi ve değerlendirme
def train_and_evaluate(model_class, env_name, timesteps, num_trials=5, render=False):
    rewards = []
    env = create_env(env_name, render_mode=render)

    # Model oluşturma
    model = model_class("MlpPolicy", env, verbose=1)

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
    TIMESTEPS = 1000  # Her trial için adım sayısı
    NUM_TRIALS = 4
    RENDER = False  # Render aktif etmek için True yapabilirsiniz

    # A2C
    print("\n--- A2C Eğitim ve Değerlendirme ---")
    a2c_model, a2c_rewards, a2c_time = train_and_evaluate(A2C, ENV_NAME, TIMESTEPS, NUM_TRIALS, render=RENDER)
    plot_algorithm_results("A2C", a2c_rewards, a2c_time)
