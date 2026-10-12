# Submitted report figures

The PNG preserves Figure 10, the Drift matching explanation, from the submitted
`Twinkl Final Report (AIS PT07 Group 5)-1.pdf`. The source PDF SHA-256,
one-based PDF pages, printed report pages, crop bounds, extraction commands,
captions, descriptions and image hashes are recorded in `metadata.json`.
The submitted PDF contains a vector version of this figure. The crop was
rendered directly from that PDF at 300 dpi using Poppler `pdftoppm`; it includes
the entire figure and legend and excludes the caption. The app renders the
caption once as HTML and links each image to its full-size asset.

To reproduce a crop, set `REPORT_PDF` to the submitted PDF and
`REPORT_ASSET_DIR` to this directory, then run the corresponding
`figures.<key>.extraction.command` in `metadata.json`. Verify the source hash
before extracting and inspect the resulting figure for clipped labels or
legends. Poppler versions may produce different raster hashes; the current
file was rendered with Poppler 26.04.0.

The application needs only these packaged static files. It never opens the
submitted PDF or reads a user's Downloads directory.
