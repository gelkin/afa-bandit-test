package ru.ifmo.ctddev.mazin.AFA;

import java.awt.Color;
import java.util.List;
import java.util.Map;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;


public class LineChart extends ApplicationFrame {

    public LineChart(List<Pair<Map<Integer, Double>, String>> dataList, final String title, String xAxis, String yAxis) {
        super(title);

        final XYDataset dataset = createDataset(dataList);
        final JFreeChart chart = createChart(dataset, title, xAxis, yAxis);
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(chartPanel);
    }

    private XYDataset createDataset(List<Pair<Map<Integer, Double>, String>> datasets) {

        final XYSeriesCollection dataset = new XYSeriesCollection();

        for (Pair<Map<Integer, Double>, String> pair : datasets) {
            Map<Integer, Double> points = pair.first;
            String name = pair.second;

            final XYSeries series = new XYSeries(name);
            for (Map.Entry<Integer, Double> point : points.entrySet()) {
                series.add(point.getKey(), point.getValue());
            }
            dataset.addSeries(series);
        }

        return dataset;
    }

    private JFreeChart createChart(final XYDataset dataset, String title, String xAxis, String yAxis) {

        // create the chart...
        final JFreeChart chart = ChartFactory.createXYLineChart(
                title,      // chart title
                xAxis,                      // x axis label
                yAxis,                      // y axis label
                dataset,                    // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips
                false                     // urls
        );

        // NOW DO SOME OPTIONAL CUSTOMISATION OF THE CHART...
        chart.setBackgroundPaint(Color.white);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.lightGray);
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);

        // first
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        plot.setDataset(0, dataset);
        plot.setRenderer(0, renderer);
        renderer.setSeriesPaint(0, Color.red);
        renderer.setSeriesFillPaint(0, Color.yellow);
        renderer.setSeriesOutlinePaint(0, Color.gray);

        renderer.setSeriesPaint(1, Color.blue);
        renderer.setSeriesFillPaint(1, Color.green);
        renderer.setSeriesOutlinePaint(1, Color.gray);

        renderer.setUseOutlinePaint(true);
        renderer.setUseFillPaint(true);

        // change the auto tick unit selection to integer units only...
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRangeIncludesZero(false);
//        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        // OPTIONAL CUSTOMISATION COMPLETED.

        return chart;

    }

    public static void run(List<Pair<Map<Integer, Double>, String>> dataList) {
        LineChart.run(dataList, "Title", "X", "Y");
    }

    public static void run(List<Pair<Map<Integer, Double>, String>> dataList, String title, String xAxis, String yAxis) {
        final LineChart demo = new LineChart(dataList, title, xAxis, yAxis);
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }

}