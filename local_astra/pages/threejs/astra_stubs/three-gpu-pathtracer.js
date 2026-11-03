// SPDX-License-Identifier: Apache-2.0
// T-0210 B0 P1 (H-T210-3): an inert local stand-in for the off-origin three-gpu-pathtracer module.
// The editor imports WebGLPathTracer for View > Realistic shading. Every method does nothing, so that
// shading mode draws nothing new on this copy; the rest of the editor never calls it.

export class WebGLPathTracer {

	constructor( renderer ) {

		this.renderer = renderer;
		this.samples = 0;
		this.filterGlossyFactor = 0;

	}

	setScene() {}

	updateCamera() {}

	updateEnvironment() {}

	updateMaterials() {}

	renderSample() {}

	reset() {}

	dispose() {}

}
