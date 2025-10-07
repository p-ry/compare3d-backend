# Compare3D Backend

FastAPI + CadQuery API to compare two STEP models.
- `POST /compare` with `file_a`, `file_b` (multipart)
- Options: `autorotate` (bool), `prefer_cog` (bool), tolerances (`t_vol`, `t_mass`, `t_cog`, `t_bbox`), `density`.

Run with Docker:
docker build -t compare3d-backend .
docker run -p 8000:8000 -e CORS_ORIGINS="http://localhost:5173,https://www.pryland.com
" compare3d-backend
