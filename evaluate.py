"""
Evaluación del modelo DQN para Ultimate TicTacToe
"""
import numpy as np
import argparse
from typing import Optional
import os

from UT3_env import UT3_Env, preprocess_state, STATE_SIZE
from dqn_model import DQNAgent


def evaluate_agent(
    model_path: str,
    num_episodes: int = 100,
    verbose: bool = True,
    render: bool = False
) -> dict:
    """
    Evalúa un modelo guardado contra oponente aleatorio.
    
    Args:
        model_path: Path al modelo .pth
        num_episodes: Número de partidas a jugar
        verbose: Mostrar progreso
        render: Mostrar tablero (solo si verbose=True)
    
    Returns:
        Dict con estadísticas de evaluación
    """
    env = UT3_Env(opponent="random")
    
    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=81,
        use_dueling=True,
        use_prioritized=False,  # No necesario para evaluación
    )
    
    agent.load(model_path)
    agent.epsilon = 0.0  # Sin exploración
    
    if verbose:
        print(f"Modelo cargado desde: {model_path}")
        print(f"Device: {agent.device}")
        print(f"Evaluando {num_episodes} partidas...")
        print("-" * 40)
    
    # Estadísticas
    wins, losses, ties = 0, 0, 0
    total_rewards = []
    game_lengths = []
    invalid_moves = 0
    
    # Análisis de juego
    first_move_wins = 0  # Victorias cuando agente mueve primero
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = preprocess_state(obs)
        total_reward = 0
        steps = 0
        episode_invalid = 0
        
        while True:
            valid_actions = info["valid_moves"]
            if not valid_actions:
                break
            
            action = agent.act(state, valid_actions)
            
            # Verificar si la acción es válida (debugging)
            if action not in valid_actions:
                episode_invalid += 1
                action = np.random.choice(valid_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            state = preprocess_state(obs)
            total_reward += reward
            steps += 1
            
            if render and verbose:
                env.render()
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        game_lengths.append(steps)
        invalid_moves += episode_invalid
        
        # Determinar resultado
        winner = info.get("winner")
        if winner == "X":
            wins += 1
            first_move_wins += 1
        elif winner == "O":
            losses += 1
        else:
            ties += 1
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Partida {episode + 1}: {winner or 'Empate'} | "
                  f"Reward: {total_reward:.2f} | Steps: {steps}")
    
    # Calcular métricas
    results = {
        "total_games": num_episodes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / num_episodes,
        "loss_rate": losses / num_episodes,
        "tie_rate": ties / num_episodes,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_game_length": np.mean(game_lengths),
        "invalid_moves": invalid_moves,
    }
    
    if verbose:
        print("\n" + "=" * 40)
        print("RESULTADOS DE EVALUACIÓN")
        print("=" * 40)
        print(f"Partidas jugadas: {num_episodes}")
        print(f"Victorias:  {wins:4d} ({results['win_rate']:.1%})")
        print(f"Derrotas:   {losses:4d} ({results['loss_rate']:.1%})")
        print(f"Empates:    {ties:4d} ({results['tie_rate']:.1%})")
        print(f"Reward promedio: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"Duración promedio: {results['avg_game_length']:.1f} pasos")
        if invalid_moves > 0:
            print(f"⚠ Movimientos inválidos: {invalid_moves}")
    
    return results


def compare_models(model_paths: list, num_episodes: int = 100):
    """Compara múltiples modelos."""
    print("\nCOMPARACIÓN DE MODELOS")
    print("=" * 60)
    
    results = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"⚠ Modelo no encontrado: {path}")
            continue
        
        name = os.path.basename(path)
        print(f"\nEvaluando: {name}")
        
        result = evaluate_agent(path, num_episodes, verbose=False)
        result["model"] = name
        results.append(result)
    
    # Mostrar tabla comparativa
    print("\n" + "-" * 60)
    print(f"{'Modelo':<30} {'Win%':>8} {'Loss%':>8} {'Tie%':>8} {'Reward':>10}")
    print("-" * 60)
    
    for r in sorted(results, key=lambda x: x["win_rate"], reverse=True):
        print(f"{r['model']:<30} {r['win_rate']:>7.1%} {r['loss_rate']:>7.1%} "
              f"{r['tie_rate']:>7.1%} {r['avg_reward']:>10.3f}")
    
    return results


def play_demo(model_path: str):
    """Muestra una partida completa con visualización."""
    env = UT3_Env(opponent="random")
    
    agent = DQNAgent(state_size=STATE_SIZE, action_size=81)
    agent.load(model_path)
    agent.epsilon = 0.0
    
    print("=" * 40)
    print("DEMO: Agente vs Oponente Aleatorio")
    print("=" * 40)
    
    obs, info = env.reset()
    state = preprocess_state(obs)
    move_num = 0
    
    while True:
        valid_actions = info["valid_moves"]
        if not valid_actions:
            break
        
        action = agent.act(state, valid_actions)
        
        # Mostrar movimiento
        br = action // 27
        bc = (action % 27) // 9
        cr = (action % 9) // 3
        cc = action % 3
        
        print(f"\nMovimiento {move_num + 1}: Tablero ({br},{bc}), Celda ({cr},{cc})")
        
        obs, reward, terminated, truncated, info = env.step(action)
        state = preprocess_state(obs)
        
        env.render()
        print(f"Reward: {reward:.2f}")
        
        move_num += 1
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 40)
    winner = info.get("winner")
    if winner == "X":
        print("🎉 ¡El agente (X) ganó!")
    elif winner == "O":
        print("😔 El oponente (O) ganó")
    else:
        print("🤝 Empate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo DQN")
    parser.add_argument("model", type=str, help="Path al modelo .pth")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--demo", action="store_true", help="Mostrar una partida")
    parser.add_argument("--compare", nargs="+", help="Comparar múltiples modelos")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare, args.episodes)
    elif args.demo:
        play_demo(args.model)
    else:
        evaluate_agent(args.model, args.episodes)
