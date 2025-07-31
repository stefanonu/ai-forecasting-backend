from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import io
import json
from typing import List, Dict, Any, Optional
import warnings
import os
warnings.filterwarnings('ignore')

app = FastAPI(title="AI Forecasting API", version="2.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_column: str
    value_column: str
    forecast_periods: int = 12
    model_type: str = "neuralprophet"

class ColumnAnalysisRequest(BaseModel):
    columns: List[str]
    sample_data: List[Dict[str, Any]]

class ForecastResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    historical_data: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    error: Optional[str] = None

class ColumnSuggestions(BaseModel):
    dateColumn: str
    valueColumns: List[str]
    reasoning: Dict[str, str]
    method: str
    confidence: float

def analyze_columns_smart_rules(columns: List[str], sample_data: List[Dict[str, Any]]) -> Dict:
    """Smart rule-based column analysis - FREE and fast"""
    
    # Date column detection
    date_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'day', 'datetime']
    date_column = None
    date_confidence = 0.0
    
    for col in columns:
        col_lower = col.lower().replace('_', ' ').replace('-', ' ')
        for keyword in date_keywords:
            if keyword in col_lower:
                date_column = col
                date_confidence = 0.9 if keyword in ['date', 'time'] else 0.7
                break
        if date_column:
            break
    
    # Value columns detection
    value_keywords = {
        'sales': ['sales', 'revenue', 'income', 'total_sales', 'gross_sales'],
        'profit': ['profit', 'margin', 'earnings', 'net_profit', 'net_income'],
        'quantity': ['quantity', 'units', 'count', 'volume', 'units_sold'],
        'price': ['price', 'cost', 'amount', 'value', 'total']
    }
    
    value_columns = []
    value_confidence = 0.0
    
    # First pass: find columns with business keywords
    for category, keywords in value_keywords.items():
        for col in columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            for keyword in keywords:
                if keyword in col_lower and col not in value_columns and col != date_column:
                    value_columns.append(col)
                    if category in ['sales', 'profit']:
                        value_confidence = max(value_confidence, 0.9)
                    else:
                        value_confidence = max(value_confidence, 0.7)
                    break
    
    # Second pass: find numeric columns if needed
    if len(value_columns) < 2:
        for col in columns:
            if col not in value_columns and col != date_column:
                try:
                    numeric_count = 0
                    total_samples = 0
                    for row in sample_data[:5]:
                        if row.get(col) is not None:
                            total_samples += 1
                            val = str(row[col]).strip().replace(',', '').replace('$', '')
                            try:
                                float(val)
                                numeric_count += 1
                            except:
                                pass
                    
                    if total_samples > 0 and numeric_count / total_samples >= 0.6:
                        value_columns.append(col)
                        if value_confidence < 0.5:
                            value_confidence = 0.6
                except:
                    pass
    
    # Calculate overall confidence
    overall_confidence = (date_confidence + value_confidence) / 2 if date_column and value_columns else 0.3
    
    return {
        "dateColumn": date_column or (columns[0] if columns else ""),
        "valueColumns": value_columns[:3] or [col for col in columns if col != date_column][:3],
        "reasoning": {
            "dateColumn": f"Found date column '{date_column}' using keyword matching" if date_column else "No clear date column found",
            "primaryValue": f"Found {len(value_columns)} business metric columns" if value_columns else "Using numeric columns as fallback",
            "alternatives": f"Detected {len([c for c in columns if c != date_column and c not in value_columns])} other potential columns"
        },
        "method": "rules",
        "confidence": overall_confidence
    }

def analyze_columns_hybrid(columns: List[str], sample_data: List[Dict[str, Any]]) -> ColumnSuggestions:
    """Hybrid approach: Rules first, LLM only if needed"""
    
    # Always try rule-based first
    rule_result = analyze_columns_smart_rules(columns, sample_data)
    
    # For now, just return rule-based results
    # LLM integration can be added later if needed
    
    return ColumnSuggestions(
        dateColumn=rule_result["dateColumn"],
        valueColumns=rule_result["valueColumns"],
        reasoning=rule_result["reasoning"],
        method=rule_result["method"],
        confidence=rule_result["confidence"]
    )

def prepare_data_for_prophet(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Prepare data in NeuralProphet format"""
    
    prophet_df = df[[date_col, value_col]].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Convert date column
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Convert value column to numeric
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    
    # Remove rows with NaN values
    prophet_df = prophet_df.dropna()
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    return prophet_df

@app.post("/analyze-columns", response_model=ColumnSuggestions)
async def analyze_columns(request: ColumnAnalysisRequest):
    """Analyze uploaded data columns using hybrid approach"""
    try:
        suggestions = analyze_columns_hybrid(request.columns, request.sample_data)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column analysis failed: {str(e)}")

@app.post("/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    """Generate forecasts using NeuralProphet"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Prepare data for NeuralProphet
        prophet_df = prepare_data_for_prophet(df, request.date_column, request.value_column)
        
        if len(prophet_df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points for forecasting")
        
        # Initialize NeuralProphet
        model = NeuralProphet(
            n_epochs=30,
            batch_size=min(32, len(prophet_df) // 2),
            learning_rate=0.01,
            n_forecasts=1,
            n_lags=min(5, len(prophet_df) // 4),
            yearly_seasonality=True if len(prophet_df) > 365 else False,
            weekly_seasonality=True if len(prophet_df) > 14 else False,
            daily_seasonality=False,
            trend_global_local="global",
            season_global_local="global",
            normalize="standardize"
        )
        
        # Fit the model
        model.fit(prophet_df, freq='D')
        
        # Create future dataframe
        future = model.make_future_dataframe(prophet_df, periods=request.forecast_periods)
        
        # Generate predictions
        forecast = model.predict(future)
        
        # Prepare historical data
        historical_data = []
        for _, row in prophet_df.iterrows():
            historical_data.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "actual": float(row['y']),
                "forecast": None,
                "type": "historical"
            })
        
        # Prepare predictions
        predictions = []
        forecast_start_idx = len(prophet_df)
        
        for i in range(forecast_start_idx, len(forecast)):
            row = forecast.iloc[i]
            predictions.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "actual": None,
                "forecast": max(0, float(row['yhat1'])),
                "type": "forecast"
            })
        
        # Model info
        model_info = {
            "model_type": request.model_type,
            "training_data_points": len(prophet_df),
            "forecast_periods": request.forecast_periods,
            "date_range": {
                "start": prophet_df['ds'].min().strftime('%Y-%m-%d'),
                "end": prophet_df['ds'].max().strftime('%Y-%m-%d')
            },
            "avg_forecast": np.mean([p["forecast"] for p in predictions if p["forecast"] is not None])
        }
        
        return ForecastResponse(
            success=True,
            predictions=predictions,
            historical_data=historical_data,
            model_info=model_info
        )
        
    except Exception as e:
        return ForecastResponse(
            success=False,
            predictions=[],
            historical_data=[],
            model_info={},
            error=str(e)
        )

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and parse CSV file"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        data = df.head(1000).to_dict('records')
        columns = df.columns.tolist()
        
        return {
            "success": True,
            "data": data,
            "columns": columns,
            "total_rows": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "AI Forecasting API with Hybrid Column Analysis",
        "version": "2.0.0",
        "status": "deployed",
        "features": {
            "column_analysis": "Smart rules-based detection",
            "forecasting": "NeuralProphet",
            "cost": "Free"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
