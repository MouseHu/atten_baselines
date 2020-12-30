import numpy as np
from attn_toy.memory.episodic_memory import EpisodicMemory
import pickle
import os
from copy import deepcopy
from attn_toy.env.fourrooms import FourroomsNorender
from attn_toy.env.fourrooms_coin_norender import FourroomsCoinNorender
from attn_toy.env.wrappers import ImageInputWarpper
import cv2

def value_iteration(env, gamma=0.95, buffer_size=2000, filedir=None,showQ=True):
    """
    算出每个obs(state)的Q函数
    """
    if filedir is not None:
        filename = os.path.join(filedir, "replay_buffer.pkl")
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                print("direct loading")
                replay_buffer = pickle.load(file)
                replay_buffer.gamma = 1
                return replay_buffer

    env.reset()
    #for showing Q
    if showQ:
        img=env.render_huge()

    cv2.imwrite('test0.jpg',env.render())
    #reset函数只能调用一次
    state_zero=env.state
    
    values={}
    transition={}
    rewards={}
    dones={}
    Q={}
    #fix goal_n,find all possible states that have the same goal，current_step is constantly 0
    all_states=[]
    possible_positions=list(range(state_zero.num_pos))
    possible_positions.remove(state_zero.goal_n)

    if not env.have_coin:
        #for env without coin
        for i in possible_positions:
            state_zero.position_n=i
            all_states.append(deepcopy(state_zero))
    else:
        for i in possible_positions:
            state_zero.position_n=i
            if len(list(state_zero.coin_dict.items()))==1:
                for k,v in state_zero.coin_dict.items():
                    if state_zero.position_n!=k:
                        state_zero.coin_dict[k]=(state_zero.coin_dict[k][0],True)
                        all_states.append(deepcopy(state_zero))

                    state_zero.coin_dict[k]=(state_zero.coin_dict[k][0],False)
                    all_states.append(deepcopy(state_zero))
            #FIXME
            else:
                raise NotImplementedError("value iteration for more than one coin is not implemented{}")
    
    #先算出所有的state和transition
    for state in all_states:
        state_t=state.to_tuple()
        
        values[state_t]=0
        rewards[state_t]=[0]*(env.action_space.n)
        transition[state_t]=[0]*(env.action_space.n)
        dones[state_t]=[0]*(env.action_space.n)
        
        for a in range(env.action_space.n):
            env.load(deepcopy(state))
            if env.state.done:
                print((env.state is state))
                print("what happened?")
                print(state_t)
            
            
            obs_tp1, reward, done, info = env.step(a)
            #assume infinite horizon 
            env.state.current_steps=0
            transition[state_t][a]=deepcopy(env.state)
            rewards[state_t][a]=reward
            dones[state_t][a]=done

    #ieratively calc Q value
    for _ in range((300)):
        for s in all_states:
            Q[s.to_tuple()]=[0]*4
            # q = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                Q[s.to_tuple()][a] = rewards[s.to_tuple()][a]
                if not dones[s.to_tuple()][a]:
                    Q[s.to_tuple()][a] += gamma * values[transition[s.to_tuple()][a].to_tuple()]

                Q[s.to_tuple()][a]=round(Q[s.to_tuple()][a],3)
            values[s.to_tuple()] = np.max(Q[s.to_tuple()])
    #print Q in screen
    if showQ:
        #WARNING:only correct when there is 1 coin
        #img=env.render_huge()
        
        for s in all_states:
            font = cv2.FONT_HERSHEY_PLAIN
            value=values[s.to_tuple()]
            cell=env.tocell[s.position_n]
            rel_pos=40 if list(s.coin_dict.items())[0][1][1] else 10
            cv2.putText(img, str(value),(cell[1]*50+10,cell[0]*50+rel_pos),font,0.7,(0,0,0),1)

        cv2.imwrite('value_iteration.jpg',img)
    #print(Q)

    replay_buffer = EpisodicMemory(buffer_size, env.observation_space.shape, env.action_space.n,gamma=gamma)
    #pust value_iteration results in replay buffer
    for s in all_states:
        s_t=s.to_tuple()
        for a in range(env.action_space.n):
            replay_buffer.store(env.render_state(s), a, rewards[s_t][a], dones[s_t][a],Q[s_t][a],\
                                env.render_state(transition[s_t][a]))
    
    print("value iteration finished")
    # print(env.color)
    # assert len(obses) == len(values)
    # value_dict = {obs.astype(np.uint8).data.tobytes(): 0.99 ** (100 - value) for obs, value in zip(obses, values)}
    if filedir is not None:
        file = os.path.join(filedir, "replay_buffer.pkl")
        if not os.path.exists(file):
            if not os.path.exists(filedir):
                os.makedirs(filedir, exist_ok=True)
            with open(os.path.join(filedir, "replay_buffer.pkl"), "wb") as file:
                print("saving replaybuffer")
                pickle.dump(replay_buffer, file)

    return replay_buffer


if __name__=='__main__':
    env=ImageInputWarpper(FourroomsCoinNorender())
    value_iteration(env)