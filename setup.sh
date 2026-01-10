cd ~/advtex_init_align
conda activate adv_align
CODE_ROOT=$PWD
export TEX_INIT_DIR=${CODE_ROOT}/advtex_init_align/tex_init
export THIRDPARTY_DIR=${CODE_ROOT}/third_parties
export SCENE_ID=scene_04
export PYTHONPATH=${CODE_ROOT}:$PYTHONPATH