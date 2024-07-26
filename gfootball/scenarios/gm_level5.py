# Level 5: easy 7v7 match, both in formation 2-1-2-1

from . import *

def build_scenario(builder):
  builder.config().game_duration = 3000
  builder.config().second_half = 1500
  builder.config().right_team_difficulty = 0.05
  builder.config().left_team_difficulty = 0.05
  builder.config().deterministic = False
  if builder.EpisodeNumber() % 2 == 0:
    first_team = Team.e_Left
    second_team = Team.e_Right
  else:
    first_team = Team.e_Right
    second_team = Team.e_Left
  builder.SetTeam(first_team)
  builder.AddPlayer(-1.0,  0.00, e_PlayerRole_GK, controllable=False)
  builder.AddPlayer( 0.0,  0.02, e_PlayerRole_CM) 
  builder.AddPlayer( 0.0, -0.02, e_PlayerRole_CF) 
  builder.AddPlayer(-0.8, -0.14, e_PlayerRole_CB)
  builder.AddPlayer(-0.8,  0.14, e_PlayerRole_CB)
  builder.AddPlayer(-0.4, -0.24, e_PlayerRole_RM)
  builder.AddPlayer(-0.4,  0.24, e_PlayerRole_LM)
  builder.SetTeam(second_team)
  builder.AddPlayer(-1.0,  0.00, e_PlayerRole_GK, controllable=False)
  builder.AddPlayer(-0.8, -0.14, e_PlayerRole_CB)
  builder.AddPlayer(-0.8,  0.14, e_PlayerRole_CB)
  builder.AddPlayer(-0.4, -0.24, e_PlayerRole_RM)
  builder.AddPlayer(-0.4,  0.24, e_PlayerRole_LM)
  builder.AddPlayer(-0.6,  0.00, e_PlayerRole_CM)
  builder.AddPlayer(-0.2,  0.00, e_PlayerRole_CF)
  