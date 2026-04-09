"""
Entrenamiento DQN para Ultimate TicTacToe
Mejoras:
- Logging detallado
- Checkpoints automáticos
- Métricas de evaluación durante entrenamiento
- Configuración centralizada
"""
import numpy as np
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import json

from UT3_env import UT3_Env, preprocess_state, STATE_SIZE
from dqn_model import DQNAgent


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    # Episodios
    episodes: int = 50000
    max_steps_per_episode: int = 200
    
    # Modelo
    use_dueling: bool = True
    use_prioritized: bool = True
    
    # Hiperparámetros
    learning_rate: float = 0.0001
    gamma: float = 0.99
    batch_size: int = 64
    memory_size: int = 100000
    target_update_freq: int = 1000
    
    # Epsilon
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30000
    
    # Logging y checkpoints
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    eval_episodes: int = 20
    
    # Paths
    save_dir: str = "models"
    log_dir: str = "logs"


def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = 20,
    verbose: bool = False
) -> dict:
    """
    Evalúa el agente contra oponente aleatorio.
    
    Returns:
        Dict con win_rate, loss_rate, tie_rate, avg_reward
    """
    env = UT3_Env(opponent="random")
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Sin exploración
    
    wins, losses, ties = 0, 0, 0
    total_rewards = []
    game_lengths = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        state = preprocess_state(obs)
        total_reward = 0
        steps = 0
        
        while True:
            valid_actions = info["valid_moves"]
            if not valid_actions:
                break
            
            action = agent.act(state, valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            state = preprocess_state(obs)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        game_lengths.append(steps)
        
        # Determinar resultado
        winner = info.get("winner")
        if winner == "X":
            wins += 1
        elif winner == "O":
            losses += 1
        else:
            ties += 1
    
    agent.epsilon = original_epsilon
    
    return {
        "win_rate": wins / num_episodes,
        "loss_rate": losses / num_episodes,
        "tie_rate": ties / num_episodes,
        "avg_reward": np.mean(total_rewards),
        "avg_length": np.mean(game_lengths),
    }


def train(config: TrainingConfig, model_path: Optional[str] = None):
    """
    Entrena el agente DQN.
    
    Args:
        config: Configuración de entrenamiento
        model_path: Path para cargar modelo existente (opcional)
    """
    # Crear directorios
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Timestamp para identificar esta sesión
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inicializar entorno y agente
    env = UT3_Env(opponent="random")
    
    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=81,
        use_dueling=config.use_dueling,
        use_prioritized=config.use_prioritized,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_steps=config.epsilon_decay_steps,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_freq,
    )
    
    if model_path:
        print(f"Cargando modelo desde {model_path}")
        agent.load(model_path)
    
    print(f"Device: {agent.device}")
    print(f"Parámetros del modelo: {sum(p.numel() for p in agent.model.parameters()):,}")
    print(f"Configuración: {config}")
    print("-" * 60)
    
    # Métricas
    episode_rewards = []
    episode_lengths = []
    wins, losses, ties = 0, 0, 0
    best_win_rate = 0.0
    
    # Log file
    log_path = os.path.join(config.log_dir, f"training_{timestamp}.json")
    training_log = []
    
    start_time = time.time()
    
    for episode in range(config.episodes):
        obs, info = env.reset()
        state = preprocess_state(obs)
        total_reward = 0
        steps = 0
        
        while steps < config.max_steps_per_episode:
            valid_actions = info["valid_moves"]
            if not valid_actions:
                break
            
            action = agent.act(state, valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_state(obs)
            
            agent.remember(state, action, reward, next_state, terminated)
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Tracking resultados
        winner = info.get("winner")
        if winner == "X":
            wins += 1
        elif winner == "O":
            losses += 1
        else:
            ties += 1
        
        # Logging periódico
        if (episode + 1) % config.log_interval == 0:
            recent_rewards = episode_rewards[-config.log_interval:]
            recent_lengths = episode_lengths[-config.log_interval:]
            
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            
            print(
                f"Ep {episode + 1:5d}/{config.episodes} | "
                f"Reward: {np.mean(recent_rewards):6.3f} | "
                f"Len: {np.mean(recent_lengths):5.1f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"W/L/T: {wins}/{losses}/{ties} | "
                f"{eps_per_sec:.1f} ep/s"
            )
            
            wins, losses, ties = 0, 0, 0
        
        # Evaluación periódica
        if (episode + 1) % config.eval_interval == 0:
            eval_results = evaluate_agent(agent, config.eval_episodes)
            
            print(f"  → Eval: Win {eval_results['win_rate']:.1%} | "
                  f"Loss {eval_results['loss_rate']:.1%} | "
                  f"Tie {eval_results['tie_rate']:.1%}")
            
            training_log.append({
                "episode": episode + 1,
                "eval": eval_results,
                "epsilon": agent.epsilon,
                "avg_reward": float(np.mean(episode_rewards[-config.eval_interval:])),
            })
            
            # Guardar mejor modelo
            if eval_results["win_rate"] > best_win_rate:
                best_win_rate = eval_results["win_rate"]
                best_path = os.path.join(config.save_dir, "best_model.pth")
                agent.save(best_path)
                print(f"  → Nuevo mejor modelo guardado (win_rate: {best_win_rate:.1%})")
        
        # Checkpoint periódico
        if (episode + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.save_dir, f"checkpoint_{episode + 1}.pth"
            )
            agent.save(checkpoint_path)
            
            # Guardar log
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
    
    # Guardar modelo final
    final_path = os.path.join(config.save_dir, f"final_{timestamp}.pth")
    agent.save(final_path)
    
    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time/3600:.2f} horas")
    print(f"Mejor win rate: {best_win_rate:.1%}")
    print(f"Modelo final guardado en: {final_path}")
    
    return agent, training_log


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar DQN para Ultimate TicTacToe")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--load", type=str, default=None, help="Path al modelo para continuar")
    parser.add_argument("--no-dueling", action="store_true")
    parser.add_argument("--no-prioritized", action="store_true")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        episodes=args.episodes,
        learning_rate=args.lr,
        use_dueling=not args.no_dueling,
        use_prioritized=not args.no_prioritized,
    )
    
    train(config, model_path=args.load)
