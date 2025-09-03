#!/bin/bash

# A100 Server Data Transfer Script
# Server info: 147.46.91.57:30590

SERVER_HOST="147.46.91.57"
SERVER_PORT="30590"
SERVER_USER="root"
REMOTE_DIR="/root/2D_sim"

echo "================================================"
echo "  Data Transfer to A100 Server"
echo "================================================"
echo ""
echo "Server: $SERVER_USER@$SERVER_HOST:$SERVER_PORT"
echo "Target: $REMOTE_DIR"
echo ""

# Check if data file exists
DATA_FILE="data/Tdot/circles_only_integrated_trajs_Tdot.h5"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found: $DATA_FILE"
    exit 1
fi

echo "📦 Data file found: $DATA_FILE"
FILE_SIZE=$(ls -lh "$DATA_FILE" | awk '{print $5}')
echo "   Size: $FILE_SIZE"
echo ""

# Create remote directories first
echo "1. Creating remote directories..."
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "mkdir -p $REMOTE_DIR/data/Tdot"
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "mkdir -p $REMOTE_DIR/data/pointcloud/circle_envs"

# Transfer data file
echo ""
echo "2. Transferring Tdot data..."
echo "   This may take a few minutes..."
scp -P $SERVER_PORT "$DATA_FILE" \
    $SERVER_USER@$SERVER_HOST:$REMOTE_DIR/data/Tdot/

# Check if pointcloud data exists and transfer if available
if [ -d "data/pointcloud/circle_envs" ]; then
    echo ""
    echo "3. Transferring pointcloud data..."
    scp -P $SERVER_PORT -r data/pointcloud/circle_envs/* \
        $SERVER_USER@$SERVER_HOST:$REMOTE_DIR/data/pointcloud/circle_envs/
else
    echo ""
    echo "⚠️  Warning: Pointcloud data not found locally"
    echo "   The model will use random pointclouds as fallback"
fi

echo ""
echo "✅ Data transfer complete!"
echo ""
echo "Next steps on A100 server:"
echo "1. SSH to server: ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
echo "2. Clone repo: git clone https://github.com/cnt225/2D_sim.git"
echo "3. Run setup script: cd 2D_sim && bash setup_a100.sh"
echo ""