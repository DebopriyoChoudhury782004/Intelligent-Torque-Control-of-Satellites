rl\_hybrid\_attitude/
├── README.md
├── requirements.txt
├── run\_scripts/
│ ├── collect\_expert.sh
│ ├── bc\_pretrain.sh
│ ├── rl\_train.sh
│ └── evaluate.sh
├── src/
│ ├── envs/
│ │ └── attitude\_env.py
│ ├── expert/
│ │ └── expert\_pid.py
│ ├── imitation/
│ │ └── behavioral\_cloning.py
│ ├── rl/
│ │ └── train\_rl.py
│ ├── utils/
│ │ └── replay\_buffer.py
│ └── evaluate/
│ └── evaluate\_policy.py
└── outputs/
├── expert\_data.npz
├── bc\_policy.pth
└── rl\_model.zip

✅ 1. Open Command Prompt and go to your project
cd C:\\Users\\shrid\\Desktop\\Projects\\rl\_hybrid\_attitude

✅ 2. Activate your virtual environment
..venv\\Scripts\\activate

You should now see:
(.venv) C:\\Users\\shrid\\Desktop\\Projects\\rl\_hybrid\_attitude>

✅ 3. (Optional) Verify that all required packages are installed
python -c "import gymnasium, stable\_baselines3, torch, numpy; print('All good.')"

🚀 4. Run Expert Demonstration Collection
This will create outputs/expert\_data.npz.
python -m src.collect\_expert --episodes 120 --out outputs/expert\_data.npz

🧠 5. Run Behavioral Cloning (Supervised Imitation Learning)
This trains the imitation (BC) policy and saves:

outputs/bc\_policy.pth

Run:
python -m src.imitation.behavioral\_cloning --data outputs/expert\_data.npz --save outputs/bc\_policy.pth --epochs 100

🤖 6. Run PPO Reinforcement Learning (Fine-tuning on top of BC policy)
This will train PPO and save:

outputs/rl\_model.zip

Command:
python -m src.rl.train\_rl --timesteps 200000 --bc outputs/bc\_policy.pth --out outputs/rl\_model.zip

Use lower timesteps for testing:
python -m src.rl.train\_rl --timesteps 50000 --bc outputs/bc\_policy.pth --out outputs/rl\_model.zip

📊 7. Evaluate the final RL policy
Runs evaluation episodes and prints mean reward:
python -m src.evaluate.evaluate\_policy --model outputs/rl\_model.zip --episodes 50

🎉 8. (Optional) Deactivate virtual environment
deactivate

👍 FINAL SUMMARY (Copy–Paste Ready)
cd C:\\Users\\shrid\\Desktop\\Projects\\rl\_hybrid\_attitude
..venv\\Scripts\\activate

python -m src.collect\_expert --episodes 120 --out outputs/expert\_data.npz
python -m src.imitation.behavioral\_cloning --data outputs/expert\_data.npz --save outputs/bc\_policy.pth --epochs 100
python -m src.rl.train\_rl --timesteps 200000 --bc outputs/bc\_policy.pth --out outputs/rl\_model.zip
python -m src.evaluate.evaluate\_policy --model outputs/rl\_model.zip --episodes 50

Want one command to run everything (full pipeline)?
I can generate a run\_all.bat file that:

activates venv

collects expert data

runs BC

runs PPO

evaluates

saves logs

Just say “Create run\_all.bat”.

Extra: evaluating the BC (behavioral cloning) model

If you want to evaluate your saved BC PyTorch bc\_policy.pth (not the SB3 model), here's a tiny snippet you can add to src/evaluate/evaluate\_policy.py or run separately.

# quick BC eval (only if BCNet class is available in src.imitation.behavioral\_cloning)

import torch
from src.imitation.behavioral\_cloning import BCNet

def eval\*bc(bc\_path, episodes=50):
env = AttitudeDetumbleEnv()
model = BCNet()
model.load\_state\_dict(torch.load(bc\_path, map\_location="cpu"))
model.eval()
rewards = \[]
for \* in range(episodes):
obs, \_ = env.reset()
done = False
ep\_ret = 0.0
while not done:
with torch.no\_grad():
a = model(torch.from\_numpy(obs.astype(np.float32)).unsqueeze(0)).squeeze(0).numpy()
next\_obs, reward, terminated, truncated, info = env.step(a)
done = bool(terminated or truncated)
obs = next\_obs
ep\_ret += float(reward)
rewards.append(ep\_ret)
print("BC mean return:", float(np.mean(rewards)))

.\.venv\\Scripts\\activate
run_all.bat

