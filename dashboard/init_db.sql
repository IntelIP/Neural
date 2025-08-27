-- Database initialization script for Kalshi trading dashboard
-- Creates all necessary tables and indexes for tracking trading performance

-- Create database if not exists
-- Note: Run this command separately as superuser:
-- CREATE DATABASE kalshi_trading;

-- Connect to the database before running the rest
-- \c kalshi_trading;

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    agent_id VARCHAR(50) NOT NULL,
    market_ticker VARCHAR(100) NOT NULL,
    market_title TEXT,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity INTEGER NOT NULL,
    price NUMERIC(10, 4) NOT NULL,
    fill_price NUMERIC(10, 4),
    total_cost NUMERIC(12, 4) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'filled', 'cancelled', 'failed')),
    realized_pnl NUMERIC(12, 4),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    execution_time TIMESTAMP,
    order_type VARCHAR(20),
    strategy VARCHAR(50)
);

-- Create positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    position_id VARCHAR(50) UNIQUE NOT NULL,
    market_ticker VARCHAR(100) NOT NULL,
    market_title TEXT,
    side VARCHAR(10) NOT NULL CHECK (side IN ('yes', 'no')),
    quantity INTEGER NOT NULL,
    entry_price NUMERIC(10, 4) NOT NULL,
    current_price NUMERIC(10, 4),
    market_value NUMERIC(12, 4),
    unrealized_pnl NUMERIC(12, 4),
    unrealized_pnl_pct NUMERIC(8, 4),
    opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    stop_loss_price NUMERIC(10, 4),
    take_profit_price NUMERIC(10, 4),
    kelly_fraction NUMERIC(6, 4)
);

-- Create profit_loss table
CREATE TABLE IF NOT EXISTS profit_loss (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_pnl NUMERIC(12, 4) NOT NULL,
    daily_pnl NUMERIC(12, 4),
    realized_pnl NUMERIC(12, 4),
    unrealized_pnl NUMERIC(12, 4),
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    win_rate NUMERIC(6, 4),
    sharpe_ratio NUMERIC(8, 4),
    max_drawdown NUMERIC(12, 4),
    portfolio_value NUMERIC(12, 4),
    cash_balance NUMERIC(12, 4)
);

-- Create agent_status table
CREATE TABLE IF NOT EXISTS agent_status (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'stopped', 'error', 'paused')),
    last_heartbeat TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    messages_processed INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    start_time TIMESTAMP,
    stop_time TIMESTAMP,
    emergency_stop BOOLEAN DEFAULT FALSE,
    version VARCHAR(20),
    config TEXT
);

-- Create market_snapshots table for analysis
CREATE TABLE IF NOT EXISTS market_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    market_ticker VARCHAR(100) NOT NULL,
    yes_price NUMERIC(10, 4),
    no_price NUMERIC(10, 4),
    volume INTEGER,
    open_interest INTEGER,
    sentiment_score NUMERIC(6, 4),
    kelly_signal NUMERIC(6, 4)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trade_market ON trades(market_ticker);
CREATE INDEX IF NOT EXISTS idx_trade_status ON trades(status);

CREATE INDEX IF NOT EXISTS idx_position_market ON positions(market_ticker);
CREATE INDEX IF NOT EXISTS idx_position_updated ON positions(updated_at);

CREATE INDEX IF NOT EXISTS idx_pnl_timestamp ON profit_loss(timestamp);

CREATE INDEX IF NOT EXISTS idx_agent_status ON agent_status(status);
CREATE INDEX IF NOT EXISTS idx_agent_heartbeat ON agent_status(last_heartbeat);

CREATE INDEX IF NOT EXISTS idx_snapshot_timestamp_market ON market_snapshots(timestamp, market_ticker);

-- Create trigger to update positions.updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Insert initial agent records
INSERT INTO agent_status (agent_name, status, last_heartbeat)
VALUES 
    ('DataCoordinator', 'stopped', CURRENT_TIMESTAMP),
    ('StrategyAnalyst', 'stopped', CURRENT_TIMESTAMP),
    ('MarketEngineer', 'stopped', CURRENT_TIMESTAMP),
    ('TradeExecutor', 'stopped', CURRENT_TIMESTAMP),
    ('RiskManager', 'stopped', CURRENT_TIMESTAMP)
ON CONFLICT (agent_name) DO NOTHING;

-- Create a view for current portfolio summary
CREATE OR REPLACE VIEW portfolio_summary AS
SELECT 
    COUNT(DISTINCT p.position_id) as open_positions,
    SUM(p.market_value) as total_market_value,
    SUM(p.unrealized_pnl) as total_unrealized_pnl,
    AVG(p.unrealized_pnl_pct) as avg_unrealized_pnl_pct,
    (SELECT total_pnl FROM profit_loss ORDER BY timestamp DESC LIMIT 1) as latest_total_pnl,
    (SELECT cash_balance FROM profit_loss ORDER BY timestamp DESC LIMIT 1) as cash_balance
FROM positions p;

-- Create a view for recent trades
CREATE OR REPLACE VIEW recent_trades AS
SELECT 
    t.trade_id,
    t.market_ticker,
    t.market_title,
    t.side,
    t.quantity,
    t.price,
    t.realized_pnl,
    t.status,
    t.timestamp,
    t.strategy
FROM trades t
ORDER BY t.timestamp DESC
LIMIT 100;