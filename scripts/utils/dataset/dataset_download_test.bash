# BSD 3-Clause License
#
# Copyright (c) 2021, Andrej Orsula
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#!/usr/bin/env bash

owner="googleresearch"
model_names="3d_dollhouse_sofa
             dino_4
             nintendo_mario_action_figure
             android_figure_orange
             dpc_handmade_hat_brown
             olive_kids_robots_pencil_case
             bia_cordon_bleu_white_porcelain_utensil_holder_900028
             germanium_ge132
             schleich_african_black_rhino
             central_garden_flower_pot_goo_425
             grandmother
             schleich_s_bayala_unicorn_70432
             chelsea_lo_fl_rdheel_zaqrnhlefw8
             kong_puppy_teething_rubber_small_pink
             school_bus
             cole_hardware_butter_dish_square_red
             lenovo_yoga_2_11
             weisshai_great_white_shark
             cole_hardware_school_bell_solid_brass_38
             mini_fire_engine"

if [[ -d "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" ]]; then
    echo "Error: There are already some models inside "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models". Please move them to a temporary location before continuing. If you have downloaded training dataset bofere, please use 'ros2 run drl_grasping dataset_unset_train.bash' ('ros2 run drl_grasping set_dataset_train.bash' to reactivate it)."
    exit 1
fi

for model_name in $model_names; do
    if [[ ! -d "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models/$model_name" ]]; then
        model_uri="https://fuel.ignitionrobotics.org/1.0/$owner/models/$model_name"
        echo "Info: Downloading model '$model_uri'"
        ign fuel download -t model -u "$model_uri"
    fi
done
for job in $(jobs -p); do
    wait $job
    echo "Info: Model downloaded"
done
