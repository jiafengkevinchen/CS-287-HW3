var data;

var scale = d3.scaleSequential(d3.interpolateGreys)
  .domain([0,1]);

var textScale = d3.scaleThreshold()
  .domain([.3,])
  .range(['black', 'white']);

d3.json('data.json', (err, d) => {
  data = d;
  console.log(data);

  d3.select("#source")
    .selectAll('span')
    .data(data.source)
    .enter()
    .append('span')
    .attr('class', 'word')
    .text(d => d)
    .style("color", (d,i) => textScale(data.attn[0][i]))
    .style("background-color", (d,i) => scale(data.attn[0][i]));

  d3.select('#translated')
    .selectAll('span')
    .data(data.translated)
    .enter()
    .append('span')
    .attr('class', 'word')
    .text(d => d)
    .on("mouseover", (d,i) => {
      d3.select("#source")
        .selectAll(".word")
        .style('color', (d1,i1) => textScale(data.attn[i][i1]))
        .style("background-color", (d1,i1) => scale(data.attn[i][i1]));
    });
});
