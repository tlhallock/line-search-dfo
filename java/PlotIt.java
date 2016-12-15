import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.imageio.ImageIO;

public class PlotIt
{
	private static double MINF = -1;
	private static double MINT = -1;
	private static double MAXF = 10;
	private static double MAXT = 10;

	private static final int WIDTH = 1000;
	private static final int HEIGHT = 1000;

	private static final Pattern rowsPattern = Pattern.compile("rows: ([0-9]*)");
	private static final Pattern colsPattern = Pattern.compile("columns: ([0-9]*)");

	private static final class DoublePoint
	{
		double f, theta;

		public DoublePoint(double theta, double f)
		{
			this(theta, f, true);
		}

		public DoublePoint(double theta, double f, boolean update)
		{
			this.f = f;
			this.theta = theta;

			if (!update) return;
			MINF = Math.min(f - 1, MINF);
			MAXF = Math.max(f + 1, MAXF);

			MINT = Math.min(theta - 1, MINT);
			MAXT = Math.max(theta + 1, MAXT);
		}
	}

	public static Point map(DoublePoint p)
	{
		int x = (int) (0 + WIDTH *  (p.theta - MINT) / (MAXT - MINT));
		int y = (int) (0 + HEIGHT * (p.f     - MINF) / (MAXF - MINF));
		return new Point(x, y);
	}

	private static Double[] split(String line)
	{
		String[] split = line.split(" ");
		LinkedList<Double> vals = new LinkedList<>();

		for (int i = 0; i < split.length; i++)
		{
			if (split[i].trim().length() == 0)
				continue;
			vals.add(Double.parseDouble(split[i]));
		}

		return vals.toArray(new Double[0]);
	}
	
	private static void makeIt(String str) throws IOException
	{
		BufferedImage image = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
		Graphics2D createGraphics = image.createGraphics();

		LinkedList<DoublePoint> list = new LinkedList<>();

		try (BufferedReader reader = Files.newBufferedReader(Paths.get(str));)
		{
			int nrows = 0;
			int ncols = 0;

			Matcher matcher;
			String line;
			while ((line = reader.readLine()) != null)
			{
				System.out.println(line);
				
				if (line.contains("#"))
				{
					matcher = rowsPattern.matcher(line);
					if (matcher.find())
						nrows = Integer.parseInt(matcher.group(1));

					matcher = colsPattern.matcher(line);
					if (matcher.find())
						ncols = Integer.parseInt(matcher.group(1));
				}
				else
				{
					Double[] row1 = split(line);
					if (row1.length == 0)
						continue;
					if (row1.length != ncols)
						throw new RuntimeException("");
					list.add(new DoublePoint(row1[0], row1[1]));
				}
			}
			
			if (list.size() != nrows)
				throw new RuntimeException();
		}

		createGraphics.setColor(Color.red);
		for (DoublePoint dp : list)
		{
			Point p = map(dp);
			createGraphics.fillRect(p.x, p.y, WIDTH - p.x, HEIGHT - p.y);
		}

		Point p1, p2;
		createGraphics.setColor(Color.white);
		p1 = map(new DoublePoint(MINT, 0, false));
		p2 = map(new DoublePoint(MAXT, 0, false));
		createGraphics.drawLine(p1.x, p1.y, p2.x, p2.y);
		p1 = map(new DoublePoint(0, MINF, false));
		p2 = map(new DoublePoint(0, MAXF, false));
		createGraphics.drawLine(p1.x, p1.y, p2.x, p2.y);
		
		for (double v = MINF; v <= MAXF; v += (MAXF-MINF)/10)
		{
			Point p = map(new DoublePoint(0, v, false));
			createGraphics.drawString("f=" + v, p.x, p.y);
		}
		for (double v = MINT; v <= MAXT; v += (MAXT-MINT)/10)
		{
			Point p = map(new DoublePoint(v, 0, false));
			createGraphics.drawString("t=" + v, p.x, p.y);
		}

		BufferedImage oImage = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < image.getWidth(); i++)
			for (int j = 0; j < image.getHeight(); j++)
				oImage.setRGB(i, HEIGHT - j - 1, image.getRGB(i, j));

		ImageIO.write(oImage, "png", new File(str.substring(0, str.length()-4) + ".png"));
	}

	public static void main(String[] args) throws IOException
	{
		makeIt("/home/thallock/Documents/Source/matlab/daset.mat");
	}
}
