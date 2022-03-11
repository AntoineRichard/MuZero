from absl import flags
import core

NUM_SIMULATIONS                     = flags.DEFINE_integer('num_simulations',
                                                        64,
                                                        'Number of simulations.')
TD_STEPS                            = flags.DEFINE_integer('td_steps',
                                                        -1,
                                                        'Number of TD steps.')
NUM_UNROLL_STEPS                    = flags.DEFINE_integer('num_unroll_steps',
                                                        5,
                                                        'Number of unroll steps.')
ONE_MINUS_DISCOUNT                  = flags.DEFINE_float('one_minus_discount',
                                                        .003,
                                                        'One minus discount factor.')
DIRICHLET_ALPHA                     = flags.DEFINE_float('dirichlet_alpha',
                                                        .5,
                                                        'Dirichlet alpha.')
ROOT_EXPLORATION_FRACTION           = flags.DEFINE_float('root_exploration_fraction',
                                                        .25,
                                                        'Root exploration fraction.')
PB_C_BASE                           = flags.DEFINE_integer('pb_c_base',
                                                        19652,
                                                        'PB C Base.')
PB_C_INIT                           = flags.DEFINE_float('pb_c_init',
                                                        1.25,
                                                        'PB C Init.')
TEMPERATURE                         = flags.DEFINE_float('temperature',
                                                        1.,
                                                        'for softmax sampling of actions')
MAX_MOVES                           = flags.DEFINE_integer('max_moves',
                                                        9,
                                                        'maximum number of moves.')
PLAY_MAX_AFTER_MOVES                = flags.DEFINE_integer('play_max_after_moves',
                                                        -1,
                                                        'Play the argmax after this many game moves. -1 means never play argmax')
USE_SOFTMAX_FOR_ACTION_SELECTION    = flags.DEFINE_integer('use_softmax_for_action_selection',
                                                        0,
                                                        'Whether to use softmax (1) or regular histogram sampling (0).')
PRETRAINING                         = flags.DEFINE_integer('pretraining',
                                                        0,
                                                        'Do pretraining.')
PRETRAINING_TEMPERATURE             = flags.DEFINE_float('pretrain_temperature',
                                                        1.,
                                                        'for contrastive loss')
EPISODE_DIR                         = flags.DEFINE_string('episode_dir',
                                                        'actor_0',
                                                        'The path to the directory that stores episodes.')
FLAGS = flags.FLAGS

def visit_softmax_temperature(num_moves, training_steps, is_training=True):
    if not is_training:
        return 0.
    if training_steps < 500e3 * 1024 / FLAGS.batch_size:
        return 1. * FLAGS.temperature
    elif training_steps < 750e3 * 1024 / FLAGS.batch_size:
        return 0.5 * FLAGS.temperature
    else:
        return 0.25 * FLAGS.temperature

def muzero_config_from_flags(action_space_size, known_bounds=None):
    """Returns muzero's config based on command line flags."""
    return core.MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=FLAGS.max_moves,
        discount=1.0 - FLAGS.one_minus_discount,
        dirichlet_alpha=FLAGS.dirichlet_alpha,
        root_exploration_fraction=FLAGS.root_exploration_fraction,
        num_simulations=FLAGS.num_simulations,
        td_steps=FLAGS.td_steps,
        num_unroll_steps=FLAGS.num_unroll_steps,
        pb_c_base=FLAGS.pb_c_base,
        pb_c_init=FLAGS.pb_c_init,
        known_bounds=known_bounds,
        episode_dir=FLAGS.episode_dir,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        use_softmax_for_action_selection=(
            FLAGS.use_softmax_for_action_selection == 1))

def muzero_config_from_file(action_space_size, dict, known_bounds=None):
    return core.MuZeroConfig(
        action_space_size=action_space_size,
        known_bounds=known_bounds,
        **dict)