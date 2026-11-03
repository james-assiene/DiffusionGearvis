// SPDX-License-Identifier: Apache-2.0
// T-0210 B3: the start scenes of the three.js editor copy, by the name in ?seed=<name>.
// Each is null (an empty scene) or a scene in three.js's object JSON format (ObjectLoader), loaded by
// astra_state.js through the editor's own API. UUIDs are fixed so a seed is the same on every load.

const IDENTITY_AT_Y1 = [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 ];
const IDENTITY = [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ];

// crate: a Mesh "Crate", BoxGeometry 1 x 1 x 1 at (0, 1, 0), rotation 0, scale 1,
// MeshStandardMaterial (white, metalness 0, roughness 1); nothing selected.
const CRATE = {
	metadata: { version: 4.6, type: 'Object', generator: 'T-0210 astra_seeds.js' },
	geometries: [ {
		uuid: '7a1e0000-0000-4000-8000-000000000001',
		type: 'BoxGeometry',
		width: 1,
		height: 1,
		depth: 1,
		widthSegments: 1,
		heightSegments: 1,
		depthSegments: 1
	} ],
	materials: [ {
		uuid: '7a1e0000-0000-4000-8000-000000000002',
		type: 'MeshStandardMaterial',
		color: 16777215,
		roughness: 1,
		metalness: 0,
		emissive: 0
	} ],
	object: {
		uuid: '7a1e0000-0000-4000-8000-000000000003',
		type: 'Scene',
		name: 'Scene',
		layers: 1,
		matrix: IDENTITY,
		up: [ 0, 1, 0 ],
		children: [ {
			uuid: '7a1e0000-0000-4000-8000-000000000004',
			type: 'Mesh',
			name: 'Crate',
			layers: 1,
			matrix: IDENTITY_AT_Y1,
			up: [ 0, 1, 0 ],
			geometry: '7a1e0000-0000-4000-8000-000000000001',
			material: '7a1e0000-0000-4000-8000-000000000002'
		} ]
	}
};

export const SEEDS = {
	empty: null,
	crate: CRATE
};

// The editor's own scene environment per seed, as the Scene panel's Environment option sets it (null: None).
// crate: Room, so a metallic finish (M5) reflects something visible; it adds no scene object (T-0210 B-R20).
export const ENVIRONMENTS = {
	empty: null,
	crate: 'Room'
};
