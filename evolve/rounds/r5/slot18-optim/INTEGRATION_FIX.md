# Integration fix applied before scoring

genome.json listed all seven axes with correct impls but omitted the top-level `"chunks"` key the
schema requires, so validate() reported `genome missing chunk 'objective'` and all four gates
failed. Wrapped the existing object in `{"chunks": ...}`. No impl, parameter or axis selection
changed — pure format, no design choice.
