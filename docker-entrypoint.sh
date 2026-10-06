set -e

DB_DIR="$(dirname "${DJANGO_DB_PATH:-/data/db.sqlite3}")"
mkdir -p "$DB_DIR"

if ! touch "$DB_DIR/.write_test" 2>/dev/null; then
    echo "[entrypoint] ERRO: sem permissão de escrita em $DB_DIR" >&2
    exit 1
fi
rm -f "$DB_DIR/.write_test"

echo "[entrypoint] Aplicando migrations em ${DJANGO_DB_PATH:-/data/db.sqlite3}..."
python manage.py migrate --noinput

exec "$@"