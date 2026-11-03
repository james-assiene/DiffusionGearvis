// The A2UI identifiers the HUD's renderer (./engine.js, ./render.js) checks against (T-0024
// criterion 5, T-0164 D1, T-0184 B1). They are identifiers, never fetched, and this is the only
// HUD file that names a catalog id (T-0164 A7).

/** The A2UI spec version the gate emits (`local_astra/ui/__init__.A2UI_SPEC_VERSION`). */
export const A2UI_SPEC_VERSION = "v0.9.1";

/** The basic-catalog id **our gate** puts in `createSurface.catalogId`. */
export const GATE_CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json";

/** Our own v2 card catalog (T-0184 B1, H-T184-4; `local_astra.ui.a2ui_schema.CARDS_V2_CATALOG_ID`). */
export const CARDS_V2_CATALOG_ID = "gearvis:cards/v2";
