"""
DQN Mejorado para Ultimate TicTacToe
Mejoras:
- Dueling DQN architecture
- Red más profunda (256-256-128)
- Double DQN
- Mejor manejo de memoria
- Epsilon decay configurable
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Optional


class DuelingDQN(nn.Module):
    """
    Dueling DQN: separa value stream y advantage stream.
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [256, 256, 128]):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature extractor compartido
        layers = []
        in_size = state_size
        for h_size in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(in_size, h_size),
                nn.LayerNorm(h_size),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_size = h_size
        
        self.features = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[-2], hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[-2], hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_size)
        )
        
        # Inicialización de pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combinar: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class SimpleDQN(nn.Module):
    """DQN simple para comparación."""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PrioritizedReplayBuffer:
    """
    Replay buffer con prioridad simplificada.
    Prioriza experiencias con alto TD error.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priorización (0=uniforme, 1=completa)
        self.beta = beta    # Corrección de importance sampling
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Agrega experiencia con máxima prioridad."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Muestrea batch según prioridades."""
        n = len(self.buffer)
        
        if n == 0:
            return None
        
        # Calcular probabilidades
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Muestrear índices
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        
        # Calcular importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Incrementar beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extraer experiencias
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Actualiza prioridades basado en TD error."""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agente DQN con mejoras:
    - Double DQN
    - Dueling architecture (opcional)
    - Prioritized replay (opcional)
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        use_dueling: bool = True,
        use_prioritized: bool = True,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        memory_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "auto"
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon scheduling lineal
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Redes
        if use_dueling:
            self.model = DuelingDQN(state_size, action_size).to(self.device)
            self.target_model = DuelingDQN(state_size, action_size).to(self.device)
        else:
            self.model = SimpleDQN(state_size, action_size).to(self.device)
            self.target_model = SimpleDQN(state_size, action_size).to(self.device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Memory
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(memory_size)
            self.use_prioritized = True
        else:
            self.memory = deque(maxlen=memory_size)
            self.use_prioritized = False
        
        # Tracking
        self.steps = 0
        self.training_losses = []
    
    def remember(self, state, action, reward, next_state, done):
        """Guarda experiencia en memoria."""
        if self.use_prioritized:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Selecciona acción usando epsilon-greedy.
        
        Args:
            state: Estado actual
            valid_actions: Lista de acciones válidas
        
        Returns:
            Acción seleccionada
        """
        if not valid_actions:
            return 0  # Fallback
        
        # Exploración
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Explotación
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        # Máscara de acciones inválidas
        mask = np.full(self.action_size, -np.inf)
        mask[valid_actions] = q_values[valid_actions]
        
        return int(np.argmax(mask))
    
    def replay(self) -> Optional[float]:
        """
        Entrena la red con un batch de memoria.
        
        Returns:
            Loss promedio o None si no hay suficientes muestras
        """
        if len(self.memory) < self.batch_size:
            return None
        
        self.model.train()
        
        # Muestrear
        if self.use_prioritized:
            sample = self.memory.sample(self.batch_size)
            if sample is None:
                return None
            states, actions, rewards, next_states, dones, indices, weights = sample
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q actual
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: usar modelo actual para seleccionar acción, target para evaluar
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # TD errors para prioritized replay
        td_errors = (target_q - current_q).detach().cpu().numpy()
        
        # Loss con importance sampling weights
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        # Actualizar prioridades
        if self.use_prioritized:
            self.memory.update_priorities(indices, td_errors)
        
        # Actualizar epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        # Actualizar target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Copia pesos del modelo actual al target."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, path: str):
        """Guarda el modelo."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)
    
    def load(self, path: str):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.05)
        self.steps = checkpoint.get('steps', 0)


if __name__ == "__main__":
    # Test del agente
    state_size = 101
    action_size = 81
    
    agent = DQNAgent(state_size, action_size)
    print(f"Device: {agent.device}")
    print(f"Model parameters: {sum(p.numel() for p in agent.model.parameters()):,}")
    
    # Test forward pass
    test_state = np.random.randn(state_size).astype(np.float32)
    valid_actions = list(range(10))
    
    action = agent.act(test_state, valid_actions)
    print(f"Selected action: {action}")
    
    # Test memory y replay
    for _ in range(100):
        s = np.random.randn(state_size).astype(np.float32)
        a = np.random.randint(action_size)
        r = np.random.randn()
        ns = np.random.randn(state_size).astype(np.float32)
        d = np.random.random() < 0.1
        agent.remember(s, a, r, ns, d)
    
    loss = agent.replay()
    print(f"Training loss: {loss}")
