import os
import time
import json
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf

class FinancialDataDownloader:
    def __init__(self):
        # List of S&P 500 technology companies to start with
        self.tech_companies = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "AVGO", "ORCL", "CSCO",
            "AMD", "ADBE", "CRM", "INTC", "QCOM"
        ]

    def download_stock_data(self, period: str = "2y") -> pd.DataFrame:
        data = pd.DataFrame()
        for symbol in self.tech_companies:
            try:
                time.sleep(1)
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period, interval="1d")
                if not hist.empty:
                    hist['Symbol'] = symbol
                    data = pd.concat([data, hist])
                    print(f"Downloaded data for {symbol}")
                else:
                    print(f"No data available for {symbol}")
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        return data

    def get_company_info(self) -> List[Dict[str, Any]]:
        companies = []
        for symbol in self.tech_companies:
            try:
                time.sleep(1)
                stock = yf.Ticker(symbol)
                info = stock.get_info()
                if not info:
                    print(f"No info available for {symbol}")
                    continue
                company = {
                    'symbol': symbol,
                    'name': info.get('longName', info.get('shortName', '')),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'employees': info.get('fullTimeEmployees', 0),
                    'description': info.get('longBusinessSummary', ''),
                    'country': info.get('country', ''),
                    'website': info.get('website', '')
                }
                companies.append(company)
                print(f"Retrieved info for {symbol}")
            except Exception as e:
                print(f"Error getting info for {symbol}: {e}")
        return companies

    def get_financial_statements(self) -> Dict[str, List[Dict[str, Any]]]:
        statements = {
            'income_statements': [],
            'balance_sheets': [],
            'cash_flows': []
        }
        for symbol in self.tech_companies:
            try:
                time.sleep(1)
                stock = yf.Ticker(symbol)
                # Get income statement
                try:
                    income_stmt = stock.income_stmt
                    if income_stmt is not None and not income_stmt.empty:
                        for date in income_stmt.columns:
                            stmt = income_stmt[date].to_dict()
                            stmt['symbol'] = symbol
                            stmt['date'] = date.strftime('%Y-%m-%d')
                            statements['income_statements'].append(stmt)
                except Exception as e:
                    print(f"Error getting income statement for {symbol}: {e}")
                # Get balance sheet
                try:
                    balance = stock.balance_sheet
                    if balance is not None and not balance.empty:
                        for date in balance.columns:
                            stmt = balance[date].to_dict()
                            stmt['symbol'] = symbol
                            stmt['date'] = date.strftime('%Y-%m-%d')
                            statements['balance_sheets'].append(stmt)
                except Exception as e:
                    print(f"Error getting balance sheet for {symbol}: {e}")
                # Get cash flow
                try:
                    cash_flow = stock.cashflow
                    if cash_flow is not None and not cash_flow.empty:
                        for date in cash_flow.columns:
                            stmt = cash_flow[date].to_dict()
                            stmt['symbol'] = symbol
                            stmt['date'] = date.strftime('%Y-%m-%d')
                            statements['cash_flows'].append(stmt)
                except Exception as e:
                    print(f"Error getting cash flow for {symbol}: {e}")
                print(f"Retrieved financial statements for {symbol}")
            except Exception as e:
                print(f"Error getting statements for {symbol}: {e}")
        return statements

    def save_data(self, output_dir: str = "data/financial"):
        os.makedirs(output_dir, exist_ok=True)
        # Download and save stock data
        print("\nDownloading stock price data...")
        try:
            stock_data = self.download_stock_data()
            if not stock_data.empty:
                stock_data.to_csv(os.path.join(output_dir, "stock_prices.csv"))
                print(f"Saved stock prices for {len(stock_data['Symbol'].unique())} companies")
            else:
                raise Exception("No stock price data was downloaded")
        except Exception as e:
            print(f"Warning: Stock data download failed ({e}). Using existing stock_prices.csv if available.")
            stock_file = os.path.join(output_dir, "stock_prices.csv")
            if not os.path.exists(stock_file):
                raise RuntimeError("No stock price data available.")
        # Get and save company info
        print("\nDownloading company information...")
        try:
            company_info = self.get_company_info()
            if company_info:
                with open(os.path.join(output_dir, "company_info.json"), 'w') as f:
                    json.dump(company_info, f, indent=2)
                print(f"Saved information for {len(company_info)} companies")
            else:
                raise Exception("No company information was downloaded")
        except Exception as e:
            print(f"Warning: Company info download failed ({e}). Using existing company_info.json if available.")
            info_file = os.path.join(output_dir, "company_info.json")
            if not os.path.exists(info_file):
                raise RuntimeError("No company info available.")
        # Get and save financial statements
        print("\nDownloading financial statements...")
        try:
            statements = self.get_financial_statements()
            for stmt_type, data in statements.items():
                if data:
                    with open(os.path.join(output_dir, f"{stmt_type}.json"), 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved {len(data)} {stmt_type}")
                else:
                    print(f"No data available for {stmt_type}")
        except Exception as e:
            print(f"Warning: Financial statements download failed ({e}). Using existing files if available.")
            for stmt_type in ["income_statements", "balance_sheets", "cash_flows"]:
                stmt_file = os.path.join(output_dir, f"{stmt_type}.json")
                if not os.path.exists(stmt_file):
                    raise RuntimeError(f"No {stmt_type} data available.")
        print("\nData download process completed!")

if __name__ == "__main__":
    downloader = FinancialDataDownloader()
    downloader.save_data() 