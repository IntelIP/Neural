# Kalshi Trading Agent Dashboard

A real-time monitoring and control dashboard for the Kalshi Agentic Trading System.

## Features

### Real-time Monitoring
- **P&L Chart**: Interactive line chart showing profit/loss over time with trade markers
- **Trade History**: Detailed table of all executed trades with filtering and sorting
- **Active Positions**: Live view of current positions with unrealized P&L
- **Agent Status**: Real-time health monitoring of all 5 trading agents

### Control Panel
- **Emergency Stop**: Immediately halt all trading operations
- **Quick Controls**: Start, pause, and resume trading operations
- **Agent Management**: Individual agent status and control

### Performance Metrics
- Total P&L (realized + unrealized)
- Daily P&L tracking
- Win rate and trade statistics
- Sharpe ratio calculation
- Maximum drawdown tracking
- Portfolio value monitoring

## Architecture

### Components
- **Frontend**: Streamlit web application
- **Database**: PostgreSQL for persistent storage
- **Cache**: Redis for real-time data streaming
- **Integration**: Connects to existing Kalshi agent infrastructure

### Data Flow
```
Trading Agents → Redis Pub/Sub → Dashboard → PostgreSQL
                      ↓
                 Real-time UI Updates
```

## Installation

### Prerequisites
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- Python 3.10+

### Quick Start

1. **Clone the repository**
```bash
cd /path/to/kalshi/project
```

2. **Install dependencies**
```bash
pip install -r dashboard/requirements.txt
```

3. **Initialize the database**
```bash
psql -U postgres -c "CREATE DATABASE kalshi_trading;"
psql -U postgres -d kalshi_trading -f dashboard/init_db.sql
```

4. **Set environment variables**
```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/kalshi_trading
export REDIS_URL=redis://localhost:6379
```

5. **Run the dashboard**
```bash
streamlit run dashboard/app.py
```

### Docker Deployment

1. **Start all services**
```bash
cd dashboard
docker-compose up -d
```

2. **Access the dashboard**
Open browser to http://localhost:8501

## Configuration

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `ENABLE_DASHBOARD_DB`: Enable database integration in agents (set to 'true')

### Database Schema
The dashboard uses the following tables:
- `trades`: Historical trade records
- `positions`: Current active positions
- `profit_loss`: P&L snapshots over time
- `agent_status`: Agent operational status
- `market_snapshots`: Market data for analysis

## Usage

### Emergency Stop
1. Click the red **EMERGENCY STOP** button in the top-right
2. Confirm the action when prompted
3. All agents will immediately halt trading

### Monitoring Trades
- View recent trades in the left table
- Filter by status, market, or strategy
- Sort by any column
- Export data to CSV (coming soon)

### Position Management
- Monitor active positions in real-time
- View unrealized P&L for each position
- Track market value changes
- Identify profitable vs losing positions

### P&L Analysis
- Select date range using the date picker
- Zoom in/out on the chart
- View daily P&L bars below main chart
- Track performance metrics in cards above

## Integration with Agents

### Base Consumer Modification
To enable database integration in your agents:

1. Set environment variable:
```bash
export ENABLE_DASHBOARD_DB=true
```

2. The base_consumer will automatically:
- Record trades to database
- Update positions
- Report agent status
- Check for emergency stops

### Redis Channels
The dashboard subscribes to:
- `kalshi:trades` - Trade executions
- `kalshi:positions` - Position updates
- `kalshi:markets` - Market data
- `agent:status` - Agent health
- `agent:control` - Control commands

## Development

### Project Structure
```
dashboard/
├── app.py                 # Main Streamlit application
├── components/           # UI components
│   ├── profit_loss_chart.py
│   ├── trades_table.py
│   ├── positions_table.py
│   └── control_panel.py
├── data/                 # Data layer
│   ├── database.py      # PostgreSQL manager
│   ├── redis_stream.py  # Redis handler
│   └── models.py        # SQLAlchemy models
├── utils/               # Utilities
│   ├── calculations.py  # Metrics calculations
│   └── formatters.py    # Data formatting
├── docker-compose.yml   # Docker configuration
├── Dockerfile          # Dashboard container
├── init_db.sql        # Database schema
└── requirements.txt    # Python dependencies
```

### Adding New Features
1. Create component in `components/`
2. Add data model if needed in `data/models.py`
3. Update `app.py` to include new component
4. Test with Docker Compose

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test connection
psql -U postgres -h localhost -d kalshi_trading
```

### Redis Connection Issues
```bash
# Check Redis is running
docker ps | grep redis

# Test connection
redis-cli ping
```

### Dashboard Not Updating
1. Check Redis subscriptions are active
2. Verify agents are publishing to correct channels
3. Check browser console for errors
4. Restart dashboard container

## Performance Considerations

- Dashboard refreshes every 1 second by default
- Large trade histories may slow down rendering
- Use date filters to limit data range
- PostgreSQL indexes optimize query performance

## Security

- Use strong passwords for PostgreSQL
- Implement authentication for production
- Use SSL/TLS for database connections
- Restrict network access in Docker
- Audit emergency stop usage

## Future Enhancements

- [ ] Export functionality (CSV, Excel)
- [ ] Advanced charting options
- [ ] Risk analytics dashboard
- [ ] Backtesting integration
- [ ] Mobile responsive design
- [ ] WebSocket for true real-time updates
- [ ] Multi-user support with roles
- [ ] Alert system for P&L thresholds
- [ ] Trade replay functionality
- [ ] Performance comparison tools

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review agent logs for errors
3. Verify all services are running
4. Check database and Redis connectivity