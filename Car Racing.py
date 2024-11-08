# Gerekli kütüphaneleri import ediyoruz
import gymnasium as gym  # OpenAI Gym yerine Gymnasium kullanıyoruz
import numpy as np  # Sonuçların analizini kolaylaştırmak için
import matplotlib.pyplot as plt  # Grafik çizimi için
from stable_baselines3 import PPO, A2C, SAC  # Üç algoritmayı (PPO, A2C, SAC) import ediyoruz

# Mountain Car Continuous ortamını başlatıyoruz
env = gym.make('CarRacing-v2')

# Her algoritmanın ortalama ödüllerini kaydetmek için listeler oluşturuyoruz
ppo_rewards, a2c_rewards, sac_rewards = [], [], []

# Eğitim için toplam adım sayısını belirliyoruz
total_timesteps = 10000  # Bu değeri daha uzun eğitimler için arttırabilirsiniz

# PPO algoritmasını tanımlayıp eğitiyoruz
ppo_model = PPO("MlpPolicy", env, verbose=1)  # MlpPolicy, fully connected neural network kullanır

# Eğitim yaparken belirli adımlarda ortalama ödülleri kaydediyoruz
for _ in range(5):  # Örnek olarak 5 eğitimi döngüye aldık, daha fazla artırabilirsiniz
    ppo_model.learn(total_timesteps=total_timesteps)
    episode_rewards = []
    obs = env.reset()[0]  # İlk değeri alıyoruz
    for _ in range(200):
        action, _ = ppo_model.predict(obs)
        action = np.array(action, dtype=np.float32)  # action'u float32'ye dönüştürüyoruz
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        if done or truncated:  # İkisi de True olursa döngüden çıkıyoruz
            break
    ppo_rewards.append(np.sum(episode_rewards))  # Bir bölümün toplam ödülünü kaydediyoruz

# A2C algoritmasını tanımlayıp eğitiyoruz
a2c_model = A2C("MlpPolicy", env, verbose=1)

for _ in range(5):
    a2c_model.learn(total_timesteps=total_timesteps)
    episode_rewards = []
    obs = env.reset()[0]  # İlk değeri alıyoruz
    for _ in range(200):
        action, _ = a2c_model.predict(obs)
        action = np.array(action, dtype=np.float32)  # action'u float32'ye dönüştürüyoruz
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        if done or truncated:  # İkisi de True olursa döngüden çıkıyoruz
            break
    a2c_rewards.append(np.sum(episode_rewards))

# SAC algoritmasını tanımlayıp eğitiyoruz
sac_model = SAC("MlpPolicy", env, verbose=1)

for _ in range(5):
    sac_model.learn(total_timesteps=total_timesteps)
    episode_rewards = []
    obs = env.reset()[0]  # İlk değeri alıyoruz
    for _ in range(200):
        action, _ = sac_model.predict(obs)
        action = np.array(action, dtype=np.float32)  # action'u float32'ye dönüştürüyoruz
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        if done or truncated:  # İkisi de True olursa döngüden çıkıyoruz
            break
    sac_rewards.append(np.sum(episode_rewards))

# Her algoritmanın ortalama ödüllerini karşılaştırmak için grafik çiziyoruz
plt.figure(figsize=(10, 6))
plt.plot(ppo_rewards, label="PPO Rewards", marker='o')
plt.plot(a2c_rewards, label="A2C Rewards", marker='x')
plt.plot(sac_rewards, label="SAC Rewards", marker='s')

# Grafik için başlık ve etiketler ekliyoruz
plt.title("Mountain Car Continuous Problem: PPO, A2C, SAC Karşılaştırması")
plt.xlabel("Eğitim Denemeleri")
plt.ylabel("Toplam Ödül")
plt.legend()
plt.show()

# Programın kapanmasını önlemek için kullanıcıdan input alıyoruz
input("Programı kapatmak için herhangi bir tuşa basın...")
