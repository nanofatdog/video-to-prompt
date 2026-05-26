#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# 🎬 Video-to-Prompt WebUI — One-Click Installer
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/nanofatdog/video-to-prompt/main/install.sh | bash
#
# What it does:
#   1. Check dependencies (python3, ffmpeg, pip)
#   2. Clone the repo
#   3. Create virtual environment
#   4. Install Python packages
#   5. Create launch script
#   6. [Optional] Install as systemd service
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Config ──
REPO_URL="https://github.com/nanofatdog/video-to-prompt.git"
INSTALL_DIR="${HOME}/video-to-prompt"
VENV_DIR="${INSTALL_DIR}/venv"
REQUIRED_CMDS=("python3" "pip3" "ffmpeg" "ffprobe" "git")

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║   🎬 Video-to-Prompt WebUI Installer       ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check dependencies ──
echo -e "${BOLD}[1/5] Checking dependencies...${NC}"
missing=()
for cmd in "${REQUIRED_CMDS[@]}"; do
    if command -v "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $cmd ($(command -v $cmd))"
    else
        echo -e "  ${RED}✗${NC} $cmd — NOT FOUND"
        missing+=("$cmd")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Missing dependencies: ${missing[*]}${NC}"
    echo ""
    echo -e "Install them first:"
    echo -e "  Ubuntu/Debian: ${CYAN}sudo apt install python3 python3-pip python3-venv ffmpeg git${NC}"
    echo -e "  CentOS/RHEL:   ${CYAN}sudo yum install python3 python3-pip ffmpeg git${NC}"
    echo -e "  macOS:         ${CYAN}brew install python3 ffmpeg git${NC}"
    exit 1
fi
echo ""

# ── Step 2: Clone repo ──
echo -e "${BOLD}[2/5] Cloning repository...${NC}"
if [ -d "$INSTALL_DIR" ]; then
    echo -e "  ${YELLOW}⚠${NC}  Directory exists: $INSTALL_DIR"
    echo -e "  Updating existing repo..."
    cd "$INSTALL_DIR"
    git pull origin main 2>/dev/null || true
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi
echo -e "  ${GREEN}✓${NC} Repo ready at $INSTALL_DIR"
echo ""

# ── Step 3: Create virtual environment ──
echo -e "${BOLD}[3/5] Setting up Python virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
echo -e "  ${GREEN}✓${NC} Virtual environment: $VENV_DIR"
echo ""

# ── Step 4: Install packages ──
echo -e "${BOLD}[4/5] Installing Python packages...${NC}"
"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"
echo -e "  ${GREEN}✓${NC} All packages installed"
echo ""

# ── Step 5: Create launch script ──
echo -e "${BOLD}[5/5] Creating launch script...${NC}"
LAUNCH_SCRIPT="${INSTALL_DIR}/launch.sh"
cat > "$LAUNCH_SCRIPT" << 'LAUNCHER'
#!/usr/bin/env bash
# Video-to-Prompt WebUI Launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python3"
WEBUI_SCRIPT="${SCRIPT_DIR}/video_to_prompt_webui.py"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}${BOLD}🎬 Video-to-Prompt WebUI${NC}"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Parse args
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
SHARE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --share) SHARE="--share"; shift ;;
        --help|-h)
            echo "Usage: ./launch.sh [options]"
            echo "  --port PORT   Port to listen on (default: 7860)"
            echo "  --host HOST   Host to bind (default: 0.0.0.0)"
            echo "  --share       Enable Gradio public share link"
            exit 0
            ;;
        *) echo "Unknown: $1"; shift;;
    esac
done

echo -e "  🌐 http://${HOST}:${PORT}"
echo ""

exec "${VENV_PYTHON}" "$WEBUI_SCRIPT" --host "$HOST" --port "$PORT" $SHARE
LAUNCHER

chmod +x "$LAUNCH_SCRIPT"
echo -e "  ${GREEN}✓${NC} Launch script: ${LAUNCH_SCRIPT}"
echo ""

# ── Systemd service (optional) ──
if command -v systemctl &>/dev/null; then
    echo ""
    echo -e "${YELLOW}💡 Install as systemd service for auto-start on boot?${NC}"
    echo -e "   Run: ${CYAN}${INSTALL_DIR}/install-service.sh${NC}"
    
    SERVICE_SCRIPT="${INSTALL_DIR}/install-service.sh"
    cat > "$SERVICE_SCRIPT" << SERVICER
#!/usr/bin/env bash
# Install Video-to-Prompt as systemd service
set -euo pipefail

SERVICE_NAME="video-to-prompt"
INSTALL_DIR="${INSTALL_DIR}"

cat << EOF | sudo tee /etc/systemd/system/\${SERVICE_NAME}.service > /dev/null
[Unit]
Description=Video-to-Prompt WebUI
After=network.target

[Service]
Type=simple
User=\$USER
WorkingDirectory=\${INSTALL_DIR}
ExecStart=\${INSTALL_DIR}/launch.sh
Restart=on-failure
RestartSec=5
Environment="HOST=0.0.0.0"
Environment="PORT=7860"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "\${SERVICE_NAME}"
sudo systemctl start "\${SERVICE_NAME}"

echo "✅ Service installed and started!"
echo "   Status: systemctl status \${SERVICE_NAME}"
echo "   Logs:   journalctl -u \${SERVICE_NAME} -f"
SERVICER
    chmod +x "$SERVICE_SCRIPT"
fi

# ── Done ──
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║   ✅ Installation Complete!                 ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  🚀 Start:    ${CYAN}${INSTALL_DIR}/launch.sh${NC}"
echo -e "  🌐 WebUI:    ${CYAN}http://localhost:7860${NC}"
echo -e "  📁 Location: ${CYAN}${INSTALL_DIR}${NC}"
echo ""
echo -e "  Options:"
echo -e "    ${INSTALL_DIR}/launch.sh --port 8080 --share"
echo ""
