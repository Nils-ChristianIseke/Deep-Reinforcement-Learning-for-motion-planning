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
#!/usr/bin/env python3

import argparse
from drl_grasping.utils import ModelCollectionRandomizer


def main(args=None):

    model_collection_randomizer = ModelCollectionRandomizer(owner=args.owner,
                                                            collection=args.collection,
                                                            server=args.server,
                                                            server_version=args.version,
                                                            unique_cache=True,
                                                            enable_blacklisting=True)
    print("Processing all models from owner [%s]..." % args.owner)
    model_collection_randomizer.process_all_models(decimation_fraction_of_visual=args.decimate_fraction,
                                                   decimation_min_faces=args.decimate_min_faces,
                                                   decimation_max_faces=args.decimate_max_faces,
                                                   max_faces=40000,
                                                   max_vertices=None,
                                                   component_min_faces_fraction=0.1,
                                                   component_max_volume_fraction=0.35,
                                                   fix_mtl_texture_paths=True)
    print("Processing finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Process all models from a collection. ' +
                    'If local cache already contains one or more models from the owner [-o], ' +
                    'these models will be processed and [-s, -c, -v] arguments ignored.')
    parser.add_argument('-o', '--owner',
                        action='store',
                        default='GoogleResearch',
                        help='Owner of the collection')
    parser.add_argument('-c', '--collection',
                        action='store',
                        default='Google Scanned Objects',
                        help='Name of the collection')
    parser.add_argument('-s', '--server',
                        action='store',
                        default='https://fuel.ignitionrobotics.org',
                        help='URI to Ignition Fuel server where the collection resides')
    parser.add_argument('-v', '--version',
                        action='store',
                        default='1.0',
                        help='Version of the Fuel server')
    parser.add_argument('--decimate_fraction',
                        action='store',
                        default='0.25',
                        help='Fraction of faces collision geometry should have compared to visual geometry (min/max faces will be enforced)')
    parser.add_argument('--decimate_min_faces',
                        action='store',
                        default='40',
                        help='Min number of faces for collision geometry')
    parser.add_argument('--decimate_max_faces',
                        action='store',
                        default='200',
                        help='Max number of faces for collision geometry')

    args = parser.parse_args()

    main(args)
