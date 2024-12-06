import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from fpdf import FPDF
from swarm import Agent, Swarm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file.")

# Dataset location
dataset_path = "./dataset.csv"


# Define functions for agents
def data_quality(context_variables):
    data = pd.read_csv(dataset_path)
    data["revenue"] = (
        data["revenue"].str.replace("₱", "").str.replace(",", "").astype(float)
    )
    data["ad spend"] = (
        data["ad spend"].str.replace("₱", "").str.replace(",", "").astype(float)
    )
    data.to_csv("cleaned_sales_data.csv", index=False)
    return f"Data cleaned. Missing values: {data.isnull().sum().sum()}, Duplicates: {data.duplicated().sum()}"


def traffic_analysis(context_variables):
    data = pd.read_csv("cleaned_sales_data.csv")
    traffic_summary = (
        data.groupby(["source", "medium", "device_type"]).sum().reset_index()
    )
    traffic_summary.to_csv("traffic_summary.csv", index=False)
    return "Traffic analysis completed and saved to traffic_summary.csv."


def sales_performance(context_variables):
    data = pd.read_csv("cleaned_sales_data.csv")
    sales_summary = (
        data.groupby("source")
        .agg({"transactions": "sum", "revenue": "sum", "ad spend": "sum"})
        .reset_index()
    )
    sales_summary["ROI"] = (
        sales_summary["revenue"] - sales_summary["ad spend"]
    ) / sales_summary["ad spend"]
    sales_summary.to_csv("sales_performance_summary.csv", index=False)
    return "Sales performance analysis completed and saved to sales_performance_summary.csv."


def predictive_modeling(context_variables):
    data = pd.read_csv("cleaned_sales_data.csv")
    features = [
        "pageviews",
        "visits",
        "productClick",
        "addToCart",
        "checkout",
        "ad spend",
    ]
    data["ROI"] = (data["revenue"] - data["ad spend"]) / data["ad spend"]
    X = data[features]
    y = data["ROI"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create visualization: Predicted vs Actual ROI
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
    )
    plt.title("Predicted vs Actual ROI")
    plt.xlabel("Actual ROI")
    plt.ylabel("Predicted ROI")
    plt.tight_layout()
    plt.savefig("predictive_modeling.png")
    plt.close()

    return f"Model evaluation complete. MSE: {mse:.4f}, R2: {r2:.4f}"


def correlation_analysis(context_variables):
    data = pd.read_csv("cleaned_sales_data.csv")

    # Select only numeric columns for correlation analysis
    numeric_data = data.select_dtypes(include=["number"])
    correlation_matrix = numeric_data.corr()

    # Create heatmap visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

    return "Correlation analysis completed. Heatmap saved as 'correlation_heatmap.png'."


def anomaly_detection(context_variables):
    data = pd.read_csv("cleaned_sales_data.csv")

    # Define threshold for anomalies
    threshold = data["revenue"].mean() + 3 * data["revenue"].std()

    # Find anomalies
    anomalies = data[data["revenue"] > threshold]

    # Save anomalies to CSV
    anomalies.to_csv("anomalies.csv", index=False)

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data["revenue"], alpha=0.6, label="Normal Data")
    plt.scatter(
        anomalies.index, anomalies["revenue"], color="red", label="Anomalies", zorder=2
    )
    plt.axhline(
        y=threshold, color="orange", linestyle="--", label=f"Threshold: {threshold:.2f}"
    )
    plt.title("Anomaly Detection: Revenue")
    plt.xlabel("Index")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig("anomaly_detection.png")
    plt.close()

    return f"Anomaly detection completed. {len(anomalies)} anomalies detected. Visualization saved as 'anomaly_detection.png'."


def trend_analysis(context_variables):
    data = pd.read_csv("cleaned_sales_data.csv")
    data["date"] = pd.to_datetime(data["date"])
    data.set_index("date", inplace=True)

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data["revenue"], model="additive", period=12)

    # Plot decomposition
    decomposition.plot()
    plt.tight_layout()
    plt.savefig("trend_analysis.png")
    plt.close()

    return "Trend analysis completed. Decomposition plot saved as 'trend_analysis.png'."


def kpi_summary(context_variables):
    data = pd.read_csv("sales_performance_summary.csv")

    # Compute KPIs
    average_revenue = data["revenue"].mean()
    average_roi = data["ROI"].mean()
    total_transactions = data["transactions"].sum()

    # Create KPI bar chart
    kpi_data = pd.DataFrame(
        {
            "Metric": ["Average Revenue", "Average ROI", "Total Transactions"],
            "Value": [average_revenue, average_roi, total_transactions],
        }
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(data=kpi_data, x="Metric", y="Value")
    plt.title("Key Performance Indicators")
    plt.tight_layout()
    plt.savefig("kpi_summary.png")
    plt.close()

    return f"KPI summary completed. Avg Revenue: {average_revenue:.2f}, Avg ROI: {average_roi:.2f}, Total Transactions: {total_transactions}."


def audience_segmentation(context_variables):
    data = pd.read_csv("cleaned_sales_data.csv")

    # Segment by device type
    segmentation = data.groupby("device_type")["visits"].sum().reset_index()

    # Create pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(
        segmentation["visits"],
        labels=segmentation["device_type"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Audience Segmentation by Device Type")
    plt.tight_layout()
    plt.savefig("audience_segmentation.png")
    plt.close()

    return "Audience segmentation completed. Pie chart saved as 'audience_segmentation.png'."


def key_findings(context_variables):
    # Key findings
    findings = (
        "1. Traffic sources with the highest ROI are identified.\n"
        "2. Anomalies in revenue indicate potential one-off events or outliers.\n"
        "3. Seasonal trends in revenue show consistent patterns.\n"
        "4. Audience segmentation highlights device types contributing to higher visits.\n"
    )

    # Business recommendations
    recommendations = (
        "1. Increase investment in high-ROI traffic sources.\n"
        "2. Investigate anomalies to understand causes (e.g., spikes in revenue).\n"
        "3. Tailor campaigns to capitalize on seasonal trends.\n"
        "4. Optimize for device types generating the most visits (e.g., mobile-first design).\n"
    )

    # Technical challenges
    challenges = (
        "1. Inconsistent or missing data in raw sources requiring cleaning.\n"
        "2. Limited granularity in audience segmentation (e.g., region-level data missing).\n"
        "3. Difficulties in accurately modeling ROI due to outliers.\n"
    )

    # Future improvements
    improvements = (
        "1. Implement real-time anomaly detection and alerts.\n"
        "2. Integrate regional data for more granular segmentation.\n"
        "3. Enhance data collection to reduce missing or inconsistent records.\n"
        "4. Experiment with advanced machine learning models for better predictions.\n"
    )

    # Consolidate into a single string
    summary = (
        "Key Findings:\n"
        + findings
        + "\n\nBusiness Recommendations:\n"
        + recommendations
        + "\n\nTechnical Challenges:\n"
        + challenges
        + "\n\nFuture Improvements:\n"
        + improvements
    )

    # Return the summary for PDF integration
    return summary


def generate_pdf_report(context_variables):
    # Load data for visualizations
    traffic_data = pd.read_csv("traffic_summary.csv")
    sales_performance_data = pd.read_csv("sales_performance_summary.csv")

    # Initialize PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", size=24)
    pdf.cell(200, 10, txt="Detailed Sales Report", ln=True, align="C")
    pdf.ln(10)

    # Traffic Analysis Section
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Traffic Analysis", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0, 10, txt="Insights into traffic sources, mediums, and device types."
    )
    pdf.ln(5)

    # Traffic Analysis Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=traffic_data, x="medium", y="pageviews", hue="device_type")
    plt.title("Pageviews by Medium and Device Type")
    plt.xlabel("Traffic Medium")
    plt.ylabel("Pageviews")
    plt.tight_layout()
    plt.savefig("traffic_analysis.png")
    plt.close()
    pdf.image("traffic_analysis.png", x=10, y=None, w=180)

    # Sales Performance Section
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Sales Performance", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0, 10, txt="Analysis of revenue, transactions, and ROI across sources."
    )
    pdf.ln(5)

    # Sales Performance Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sales_performance_data, x="source", y="ROI")
    plt.title("ROI by Source")
    plt.xlabel("Source")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig("sales_performance.png")
    plt.close()
    pdf.image("sales_performance.png", x=10, y=None, w=180)

    # Predictive Modeling Section
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Predictive Modeling", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Evaluation of the predictive model for ROI.")
    pdf.ln(5)
    pdf.image("predictive_modeling.png", x=10, y=None, w=180)

    # Correlation Analysis Section
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Correlation Analysis", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0, 10, txt="Heatmap of variable correlations to identify key relationships."
    )
    pdf.ln(5)
    pdf.image("correlation_heatmap.png", x=10, y=None, w=180)

    # Anomaly Detection Section
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Anomaly Detection", ln=True)
    pdf.set_font("Arial", size=12)
    anomalies = pd.read_csv("anomalies.csv")
    pdf.multi_cell(
        0,
        10,
        txt=f"{len(anomalies)} anomalies detected. Details saved to 'anomalies.csv'.",
    )
    pdf.ln(5)
    pdf.image("anomaly_detection.png", x=10, y=None, w=180)

    # Trend Analysis Section
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Trend Analysis", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Decomposition of revenue trends over time.")
    pdf.ln(5)
    pdf.image("trend_analysis.png", x=10, y=None, w=180)

    # KPI Summary Section
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="KPI Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0, 10, txt="Overview of key performance indicators such as revenue and ROI."
    )
    pdf.ln(5)
    pdf.image("kpi_summary.png", x=10, y=None, w=180)

    # Audience Segmentation Section
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Audience Segmentation", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Distribution of audience across device types.")
    pdf.ln(5)
    pdf.image("audience_segmentation.png", x=10, y=None, w=180)

    # Key Findings Section
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Key Findings", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(
        0,
        10,
        txt="Key findings from the analysis, along with recommendations, challenges, and improvements.",
    )
    pdf.ln(5)
    findings = key_findings(context_variables)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, txt=findings)

    # Save PDF
    pdf.output("Sales_Report.pdf")
    return "Detailed PDF report generated and saved as 'Sales_Report.pdf'."


# Define agents
data_quality_agent = Agent(
    name="Data Quality", instructions="Clean the data.", functions=[data_quality]
)
traffic_analysis_agent = Agent(
    name="Traffic Analysis",
    instructions="Perform traffic analysis.",
    functions=[traffic_analysis],
)
sales_performance_agent = Agent(
    name="Sales Performance",
    instructions="Analyze sales performance.",
    functions=[sales_performance],
)
predictive_modeling_agent = Agent(
    name="Predictive Modeling",
    instructions="Build and evaluate a predictive model.",
    functions=[predictive_modeling],
)
correlation_analysis_agent = Agent(
    name="Correlation Analysis",
    instructions="Analyze correlations.",
    functions=[correlation_analysis],
)
anomaly_detection_agent = Agent(
    name="Anomaly Detection",
    instructions="Detect anomalies in revenue.",
    functions=[anomaly_detection],
)
trend_analysis_agent = Agent(
    name="Trend Analysis",
    instructions="Analyze time-series trends.",
    functions=[trend_analysis],
)
kpi_summary_agent = Agent(
    name="KPI Summary", instructions="Summarize key KPIs.", functions=[kpi_summary]
)
audience_segmentation_agent = Agent(
    name="Audience Segmentation",
    instructions="Segment audience by device type.",
    functions=[audience_segmentation],
)
key_findings_agent = Agent(
    name="Key Findings",
    instructions="Summarize key findings, provide business recommendations, outline technical challenges, and suggest future improvements.",
    functions=[key_findings],
)
report_generation_agent = Agent(
    name="Report Generation",
    instructions="Generate a detailed PDF report with visualizations based on analysis outputs.",
    functions=[generate_pdf_report],
)


# Create swarm client
client = Swarm()

# Sequential execution of agents
agents = [
    data_quality_agent,
    traffic_analysis_agent,
    sales_performance_agent,
    predictive_modeling_agent,
    correlation_analysis_agent,
    anomaly_detection_agent,
    trend_analysis_agent,
    kpi_summary_agent,
    audience_segmentation_agent,
    key_findings_agent,
    report_generation_agent,
]

context_variables = {}
messages = [{"role": "user", "content": "Start the analysis process."}]

for agent in agents:
    print(f"\nRunning agent: {agent.name}")
    response = client.run(
        agent=agent, messages=messages, context_variables=context_variables
    )
    context_variables = response.context_variables
    messages = response.messages
