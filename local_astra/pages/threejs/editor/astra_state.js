// SPDX-License-Identifier: Apache-2.0
// T-0210 B3: the state shim of the three.js editor copy (ours, not upstream).
//
// It reads ?seed=<name> and applies that seed from astra_seeds.js through the editor's own API
// (editor.clear, then the seed scene parsed by THREE.ObjectLoader and handed to editor.setScene; then the seed's
// environment through the editor's sceneEnvironmentChanged signal, as the Scene panel's Environment option does),
// resets the undo history and sets ready. Without ?seed= it applies nothing, reports seed: null and never sets ready
// (a restored autosave is never judged as a seed; B-R24). With ?seed= present the patched index.html (P6) skips the editor's
// IndexedDB autosave restore and its saved selection, so the seed is the whole scene on every load (B-R13).
// window.__state is a getter computed from editor.scene each time it is read:
//
//   { seed, ready, objects: [ { uuid, name, type, geometry: { type, parameters } | null, position: [x, y, z],
//     rotation_deg: [x, y, z], scale: [x, y, z], material: { type, metalness, roughness, color } | null,
//     intensity | null, parent } ], file_ops: { new, import, export }, scripts, player_started }
//
// objects are the scene's descendants in traversal order (not the scene, not the editor's own camera); position,
// rotation (Euler XYZ, degrees) and scale are world values; color is the lowercase #rrggbb of material.color;
// parent is the parent's name. file_ops counts clicks on File > New (any entry of its submenu), Open and Import
// (import), and Save and Export (any entry of its submenu) (export). scripts is the number of scripts ever added
// (the editor's scriptAdded signal), never decreasing: removing a script or clearing editor.scripts hides nothing.
// player_started counts the editor's startPlayer signal (Play) and, besides, clicks on Project > Play (capture phase)
// (B-R12, B-R22). Both signal listeners are added at priority 1, above the editor's own (priority 0): the shim counts
// before Player.js runs any user script, so a script that throws, halts the signal or deactivates it cannot skip the
// count.
//
// __state is for the task judge and the tests only; the model sees pixels. This file never touches the network
// and runs no timer of its own.

import * as THREE from 'three';

import { ENVIRONMENTS, SEEDS } from './astra_seeds.js';

const fileOps = { new: 0, import: 0, export: 0 };
let playerStarted = 0;
let scriptsAdded = 0;
const params = new URLSearchParams( window.location.search );
const seedName = params.get( 'seed' ); // null without ?seed= (B-R24)
let ready = false;

function vec( v ) {

	return [ v.x, v.y, v.z ];

}

function plain( value ) {

	// geometry parameters: numbers, booleans and strings only (shapes and paths are summarised by their type)
	const out = {};
	if ( value === undefined || value === null ) return out;
	for ( const key of Object.keys( value ) ) {

		const v = value[ key ];
		if ( typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string' ) out[ key ] = v;

	}

	return out;

}

function materialOf( object ) {

	let m = object.material;
	if ( Array.isArray( m ) ) m = m[ 0 ];
	if ( m === undefined || m === null ) return null;
	return {
		type: m.type,
		metalness: typeof m.metalness === 'number' ? m.metalness : null,
		roughness: typeof m.roughness === 'number' ? m.roughness : null,
		color: m.color !== undefined && m.color.isColor === true ? '#' + m.color.getHexString() : null
	};

}

function describe( object ) {

	const position = new THREE.Vector3();
	const quaternion = new THREE.Quaternion();
	const scale = new THREE.Vector3();
	object.updateWorldMatrix( true, false );
	object.matrixWorld.decompose( position, quaternion, scale );
	const euler = new THREE.Euler().setFromQuaternion( quaternion, 'XYZ' );
	const geometry = object.geometry;
	return {
		uuid: object.uuid,
		name: object.name,
		type: object.type,
		geometry: geometry === undefined || geometry === null ? null : { type: geometry.type, parameters: plain( geometry.parameters ) },
		position: vec( position ),
		rotation_deg: [ euler.x, euler.y, euler.z ].map( THREE.MathUtils.radToDeg ),
		scale: vec( scale ),
		material: materialOf( object ),
		intensity: object.isLight === true ? object.intensity : null,
		parent: object.parent === null ? '' : object.parent.name
	};

}

function snapshot() {

	const editor = window.editor;
	const objects = [];
	if ( editor !== undefined ) {

		editor.scene.traverse( function ( object ) {

			if ( object !== editor.scene ) objects.push( describe( object ) );

		} );

	}

	return {
		seed: seedName,
		ready: ready,
		objects: objects,
		file_ops: { ...fileOps },
		scripts: scriptsAdded,
		player_started: playerStarted
	};

}

Object.defineProperty( window, '__state', { get: snapshot, configurable: false, enumerable: false } );

// File > New / Open / Save / Import / Export, counted on click (capture phase, before the editor's own handler)

function ownText( element ) {

	return Array.from( element.childNodes ).filter( ( n ) => n.nodeType === 3 ).map( ( n ) => n.textContent ).join( '' ).trim();

}

function countFileOp( event ) {

	const editor = window.editor;
	const target = event.target;
	if ( editor === undefined || ! ( target instanceof Element ) ) return;
	const option = target.closest( '#menubar .option' );
	const menu = target.closest( '#menubar .menu' );
	if ( option === null || menu === null ) return;
	const title = menu.querySelector( '.title' );
	const key = ( k ) => editor.strings.getKey( k );
	if ( title === null || title.textContent.trim() !== key( 'menubar/file' ) ) return;
	if ( option.classList.contains( 'submenu-title' ) ) return;
	const submenu = option.parentElement === null ? null : option.parentElement.closest( '.submenu-title' );
	const label = submenu === null ? ownText( option ) : ownText( submenu );
	if ( label === key( 'menubar/file/new' ) ) fileOps.new += 1;
	else if ( label === key( 'menubar/file/open' ) || label === key( 'menubar/file/import' ) ) fileOps.import += 1;
	else if ( label === key( 'menubar/file/save' ) || label === key( 'menubar/file/export' ) ) fileOps.export += 1;

}

document.addEventListener( 'click', countFileOp, true );

// Project > Play, counted on click (capture phase, before the button's own handler; B-R22): the Project tab's App
// button while it reads Play (not Stop)

function countPlay( event ) {

	const editor = window.editor;
	const target = event.target;
	if ( editor === undefined || ! ( target instanceof Element ) ) return;
	const button = target.closest( '#sidebar #app button' );
	if ( button === null ) return;
	if ( button.textContent.trim() === editor.strings.getKey( 'sidebar/project/app/play' ) ) playerStarted += 1;

}

document.addEventListener( 'click', countPlay, true );

// the counters on the editor's signals, at priority 1 (above the editor's own listeners, which are at 0)

function addCounters() {

	const signals = window.editor.signals;
	signals.startPlayer.add( function () {

		playerStarted += 1;

	}, null, 1 );
	signals.scriptAdded.add( function () {

		scriptsAdded += 1;

	}, null, 1 );

}

// the seed

async function applySeed() {

	const editor = window.editor;
	addCounters();
	if ( seedName === null ) return; // no seed: nothing applied, never ready (B-R24)
	if ( ! Object.prototype.hasOwnProperty.call( SEEDS, seedName ) ) return; // an unknown seed never becomes ready
	const seed = SEEDS[ seedName ];
	editor.clear();
	if ( seed !== null ) {

		const scene = await new THREE.ObjectLoader().parseAsync( seed );
		editor.setScene( scene );

	}

	const environment = ENVIRONMENTS[ seedName ];
	if ( environment !== null ) {

		editor.signals.sceneEnvironmentChanged.dispatch( environment );
		editor.signals.refreshSidebarEnvironment.dispatch();

	}

	editor.deselect();
	editor.history.clear();
	editor.signals.historyChanged.dispatch();
	ready = true;

}

if ( window.editor !== undefined ) {

	applySeed();

} else {

	window.addEventListener( 'load', applySeed, { once: true } );

}
