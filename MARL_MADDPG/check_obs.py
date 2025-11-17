import gymnasium as gym
try:
    import gymnasium_robotics
    print(f"gymnasium_robotics version: {gymnasium_robotics.__version__}")
except ImportError:
    print("gymnasium_robotics not found, proceeding...")

print(f"Gymnasium version: {gym.__version__}")

# 建立環境
try:
    env = gym.make(
        "Ant-v5", 
        # 試著明確關閉 contact forces，看看維度是否會改變
        use_contact_forces=False, 
        exclude_current_positions_from_observation=False
    )
    print("Created Ant-v5 with `use_contact_forces=False`")
except TypeError:
    env = gym.make("Ant-v5")
    print("Created Ant-v5 with default settings (TypeError).")

obs_test, _ = env.reset()
print(f"\nActual observation shape (with use_contact_forces=False): {obs_test.shape}")

# 重新建立環境，這次使用預設值（可能包含 contact forces）
try:
    env_default = gym.make("Ant-v5")
    print("\nCreated Ant-v5 with default settings.")

    # 列印關鍵的 observation_structure
    if hasattr(env_default.unwrapped, "observation_structure"):
        print("\n--- [DEFAULT ENV] Observation Structure ---")
        print(env_default.unwrapped.observation_structure)
        print("---------------------------------------------")
    else:
        print("\nCould not find `env_default.unwrapped.observation_structure`.")

    obs_default, _ = env_default.reset()
    print(f"\nActual observation shape (default): {obs_default.shape}")
    env_default.close()

except Exception as e:
    print(f"\nFailed to create default Ant-v5 env: {e}")

env.close()