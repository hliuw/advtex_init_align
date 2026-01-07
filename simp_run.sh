#!/bin/bash
{

REPO_DIR="$1"
SCENE_ID="$2"

export PYTHONPATH=${REPO_DIR}:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate adv_align

SEED=123

N_PATCHES_H=1
N_PATCHES_W=1

# MRF related
UNARY=1
FACE_AREA_PENALTY=1e-3
DEPTH_PENALTY=-10
PERCEPT_PENALTY=-1
DUMMY_PENALTY=-15

DATA_DIR=${REPO_DIR}/dataset
EXP_DIR=${REPO_DIR}/experiments

MRF_BIN=${REPO_DIR}/advtex_init_align/tex_init/tex_init

MTL_ATLAS_SIZE=15

N_ITERS_MP=0

N_SUBDIV=0

N_MESH_SPLITS=30

MTL_RES=2048

if [ "${UNARY}" == "1" ]; then
    MRF_NAME="unary"  
    MRF_DIR_NAME="output_obj_argmax_${MTL_RES}_${MTL_RES}_500"
else
    MRF_NAME="pairwise"
    MRF_DIR_NAME="output_obj_mp_only_adj_${MTL_RES}_${MTL_RES}_500"
fi

ulimit -n 65000;
MKL_THREADING_LAYER=GNU;

printf '\n========================================\n'
printf 'Processing scene: %s\n' ${SCENE_ID}
printf 'MRF output folder: %s\n' ${MRF_DIR_NAME}
printf 'Using ALL input data (no train/test split)\n'
printf '========================================\n\n'

total_start="$(date -u +%s)"

# Use all frames - no sampling
PROCESSED_DIR=${DATA_DIR}/${SCENE_ID}

printf "=== Step 1: Splitting Mesh into Sub-meshes ===\n"

export OMP_NUM_THREADS=10 && \
${MRF_BIN} \
--data_dir ${PROCESSED_DIR} \
--debug 0 \
--debug_mesh_shape 0 \
--align_arg_max ${UNARY} \
--n_iter_mp ${N_ITERS_MP} \
--n_workers 10 \
--mtl_width ${MTL_RES} \
--mtl_height ${MTL_RES} \
--n_extrude_pixels 0 \
--iter_subdiv ${N_SUBDIV} \
--stream_type scannet \
--preprocess_split_mesh ${N_MESH_SPLITS}

printf "\n✓ Mesh splitting complete\n\n"

# Create symbolic link for stream file
ln -sf ${PROCESSED_DIR}/Recv.stream ${PROCESSED_DIR}/splitted_mesh_${N_MESH_SPLITS}/Recv.stream

printf "=== Step 2: Running MRF Texture Initialization ===\n"

export OMP_NUM_THREADS=10 && \
${MRF_BIN} \
--data_dir ${PROCESSED_DIR}/splitted_mesh_${N_MESH_SPLITS} \
--debug 1 \
--debug_mesh_shape 0 \
--align_arg_max ${UNARY} \
--n_iter_mp ${N_ITERS_MP} \
--n_workers 10 \
--mtl_width ${MTL_RES} \
--mtl_height ${MTL_RES} \
--n_extrude_pixels 0 \
--iter_subdiv ${N_SUBDIV} \
--stream_type scannet \
--unary_potential_dummy ${DUMMY_PENALTY} \
--remesh 0 \
--bin_pack_type 3 \
--n_areas_per_plate_bin_pack 500 \
--debug_mesh_shape 0 \
--conformal_map 1 \
--compact_mtl 1 \
--top_one_mtl 1 \
--penalty_face_area ${FACE_AREA_PENALTY} \
--penalty_face_cam_dist ${DEPTH_PENALTY} \
--penalty_face_v_perception_consist ${PERCEPT_PENALTY} \
--pair_potential_mp 1 \
--pair_potential_off_diag_scale_depth 0 \
--pair_potential_off_diag_scale_percept 0

printf "\n✓ MRF texture initialization complete\n\n"

# Prepare data for fast IO
printf "=== Step 3: Preparing Data for Fast IO ===\n"

python ${REPO_DIR}/advtex_init_align/data/prepare_for_scannet.py \
--stream_f_list ${PROCESSED_DIR}/Recv.stream \
--save_dir ${EXP_DIR}/${SCENE_ID}/prepare \
--stream_type scannet

printf "\n✓ Data preparation complete\n\n"

# Generate L2 averaged texture
printf "=== Step 4: Generating L2 Averaged Texture ===\n"

python ${REPO_DIR}/advtex_init_align/data/gen_avg_mtl.py \
--stream_f_list ${PROCESSED_DIR}/Recv.stream \
--obj_f_list ${PROCESSED_DIR}/splitted_mesh_${N_MESH_SPLITS}/${MRF_DIR_NAME}/TexAlign.obj \
--save_dir ${EXP_DIR}/${SCENE_ID}/avg_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE} \
--atlas_size ${MTL_ATLAS_SIZE} \
--debug_vis 0 \
--fuse 1 \
--directly_fuse 0 \
--stream_type scannet \
--scannet_data_dir ${EXP_DIR}/${SCENE_ID}/prepare/data

printf "\n✓ L2 averaged texture complete\n\n"

# Convert to optimization format
printf "=== Step 5: Converting to Optimization Format ===\n"

python ${REPO_DIR}/advtex_init_align/data/format_converter/convert_mrf_result_to_adv_tex.py \
--stream_f_list ${PROCESSED_DIR}/Recv.stream \
--obj_f_list ${PROCESSED_DIR}/splitted_mesh_${N_MESH_SPLITS}/${MRF_DIR_NAME}/fused/TexAlign.obj \
--save_dir ${EXP_DIR}/${SCENE_ID}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/fused \
--atlas_size ${MTL_ATLAS_SIZE} \
--already_single_mtl 0 \
--for_train 1 \
--stream_type scannet \
--scannet_data_dir ${EXP_DIR}/${SCENE_ID}/prepare/data

printf "\n✓ Format conversion complete\n\n"

# Run texture optimization
printf "=== Step 6: Running Texture Optimization ===\n"

python ${REPO_DIR}/advtex_init_align/tex_smooth/optim_patch_torch.py \
--seed ${SEED} \
--use_mislaign_offset 1 \
--from_scratch 0 \
--n_patches_h ${N_PATCHES_H} \
--n_patches_w ${N_PATCHES_W} \
--input_dir ${EXP_DIR}/${SCENE_ID}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/fused 

printf "\n✓ Texture optimization complete\n\n"

# Copy final results
printf "=== Step 7: Saving Final Results ===\n"

mkdir -p ${EXP_DIR}/${SCENE_ID}/final_output

cp -r ${EXP_DIR}/${SCENE_ID}/optim/${MRF_NAME}_${MTL_RES}_${MTL_RES}_atlas_${MTL_ATLAS_SIZE}/seed_${SEED}-scratch_0-offset_1_n_patch_h_${N_PATCHES_H}_w_${N_PATCHES_W}/shape/* \
      ${EXP_DIR}/${SCENE_ID}/final_output/

printf "\n✓ Final results saved to: ${EXP_DIR}/${SCENE_ID}/final_output/\n"
printf "   - TexAlign.obj\n"
printf "   - TexAlign.mtl\n"
printf "   - mtl0.png\n\n"

total_end="$(date -u +%s)"
elapsed="$(($total_end-$total_start))"
printf "========================================\n"
printf "Total time elapsed: %d seconds (%.2f minutes)\n" $elapsed $(echo "scale=2; $elapsed/60" | bc)
printf "========================================\n"

exit;
}