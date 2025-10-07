# Compare3D Backend

FastAPI + CadQuery API to compare two STEP models.
- `POST /compare` with `file_a`, `file_b` (multipart)
- Options: `autorotate` (bool), `prefer_cog` (bool), tolerances (`t_vol`, `t_mass`, `t_cog`, `t_bbox`), `density`.

Run with Docker:
