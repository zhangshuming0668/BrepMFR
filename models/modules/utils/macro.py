import numpy as np

MAIN_COMMANDS = ['SOL', 'EOS', 'box', 'prism', 'cylinder', 'cone', 'sphere']

SUB_COMMANDS = ['SOL', 'EOS', 
                'rect_slot', 'tri_slot', 'cir_slot', 'rect_psg', 'tri_psg', 'hexa_psg', 'hole', 'rect_step', 
                'tside_step', 'slant_step', 'rect_b_step', 'tri_step', 'cir_step', 'rect_b_slot', 'cir_b_slot', 
                'u_b_slot', 'rect_pkt', 'key_pkt', 'tri_pkt', 'hexa_pkt', 'o_ring', 'b_hole', 'chamfer', 'fillet']

BOX_IDX = MAIN_COMMANDS.index('box')
PRISM_IDX = MAIN_COMMANDS.index('prism')
CYLINDER_IDX = MAIN_COMMANDS.index('cylinder')
CONE_IDX = MAIN_COMMANDS.index('cone')
SPHERE_IDX = MAIN_COMMANDS.index('sphere')

RECT_SLOT_IDX = SUB_COMMANDS.index('rect_slot')
TRI_SLOT_IDX = SUB_COMMANDS.index('tri_slot')
CIR_SLOT_IDX = SUB_COMMANDS.index('cir_slot')
RECT_PSG_IDX = SUB_COMMANDS.index('rect_psg')
TRI_PSG_IDX = SUB_COMMANDS.index('tri_psg')
HEXA_PSG_IDX = SUB_COMMANDS.index('hexa_psg')
HOLE_IDX = SUB_COMMANDS.index('hole')
RECT_STEP_IDX = SUB_COMMANDS.index('rect_step')
TSIDE_STEP_IDX = SUB_COMMANDS.index('tside_step')
SLANT_STEP_IDX = SUB_COMMANDS.index('slant_step')
RECT_B_STEP_IDX = SUB_COMMANDS.index('rect_b_step')
TRI_STEP_IDX = SUB_COMMANDS.index('tri_step')
CIR_STEP_IDX = SUB_COMMANDS.index('cir_step')
RECT_B_SLOT_IDX = SUB_COMMANDS.index('rect_b_slot')
CIR_B_SLOT_IDX = SUB_COMMANDS.index('cir_b_slot')
U_B_SLOT_IDX = SUB_COMMANDS.index('u_b_slot')
RECT_PKT_IDX = SUB_COMMANDS.index('rect_pkt')
KEY_PKT_IDX = SUB_COMMANDS.index('key_pkt')
TRI_PKT_IDX = SUB_COMMANDS.index('tri_pkt')
HEXA_PKT_IDX = SUB_COMMANDS.index('hexa_pkt')
O_RING_IDX = SUB_COMMANDS.index('o_ring')
B_HOLE_IDX = SUB_COMMANDS.index('b_hole')
CHAMFER_IDX = SUB_COMMANDS.index('chamfer')
FILLET_IDX = SUB_COMMANDS.index('fillet')

SOL_IDX = MAIN_COMMANDS.index('SOL')
EOS_IDX = MAIN_COMMANDS.index('EOS')
EXT_IDX = -1

PAD_VAL = -1
N_MAIN_COMMANDS = 5
N_SUB_COMMANDS = 24
N_ARGS_MAIN = 11  # main_feature parameters: L1, L2, L3, E, Tx, Ty, Tz, Q0, Q1, Q2, Q3
N_ARGS_SUB = 12   # sub_feature parameters:  X1, Y1, Z1, X2, Y2, Z2, W, L, R, A, W1, D

MAIN_SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_ARGS_MAIN)])
MAIN_EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_ARGS_MAIN)])

SUB_SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_ARGS_SUB)])
SUB_EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_ARGS_SUB)])

                            #  L1 L2 L3  E Tx Ty Tz Q0 Q1 Q2 Q3
MAIN_CMD_ARGS_MASK = np.array([[*[0] * N_ARGS_MAIN],  # SOL
                               [*[0] * N_ARGS_MAIN],  # EOS
                               [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # box
                               [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # prism
                               [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # cylinder
                               [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # cone
                               [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # sphere
                               ])
                            
SUB_CMD_ARGS_MASK = np.array([[ *[0]*N_ARGS_SUB],  # SOL
                              [ *[0]*N_ARGS_SUB],  # EOS
                              #X1 Y1 Z1 X2 Y2 Z2 W  L  R  A  W1 D  
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # rect_slot
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # tri_slot
                              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # cir_slot
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],  # rect_psg
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],  # tri_psg
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],  # hexa_psg
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # hole
                              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],  # rect_step
                              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],  # tside_step
                              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],  # slant_step                         
                              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],  # rect_b_step
                              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],  # tri_step
                              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],  # cir_step
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # rect_b_slot                         
                              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # cir_b_slot
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # u_b_slot
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],  # rect_pkt
                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],  # key_pkt
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],  # tri_pkt
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],  # hexa_pkt
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # o_ring
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # b_hole
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # chamfer
                              [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # fillet                              
                             ])

MAX_N_MAIN = 7 # maximum number of main_feature
MAX_N_SUB = 8 # maximum number of sub_feature
# MAX_TOTAL_LEN = 15 # maximum cad sequence length
ARGS_DIM = 258 #-1-256  最大维度是258

N_ALL_SAMPLE = 1000
N_PART_SAMPLE = 210

GRID_BOUND = 500.0
GRID_SIZE = 64


