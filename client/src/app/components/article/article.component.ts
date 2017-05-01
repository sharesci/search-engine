import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { SharedService } from '../../services/shared.service.js';
import { ArticleService } from '../../services/article.service.js'
import { IArticleWrapper } from '../../entities/article-wrapper.interface.js'
import { IArticle } from '../../entities/article.interface.js'

@Component({
    templateUrl: 'src/app/components/article/article.component.html',
    styleUrls: ['src/app/components/article/article.component.css']
})

export class ArticleComponent implements OnInit {
    constructor(private _sharedService: SharedService, private _route: ActivatedRoute,
        private _articleService: ArticleService) { }

    article: IArticle = null

    ngOnInit() {
        this._articleService.getArticle(this._route.snapshot.params['id'], false)
            .map(response => <IArticleWrapper>response)
            .subscribe(
            results => this.showArticleData(results),
            error => console.log(error)
            )
    }

    showArticleData(articleWrapper: IArticleWrapper) {
        if (articleWrapper.errno == 0) {
            this.article = articleWrapper.articleJson[0];
        }
    }

    private viewPdf(download: boolean) {
        var saveAs = require('file-saver');
        this._articleService.getArticle(this.article._id, true)
            .subscribe(
            results => {
                if (download) {
                    saveAs(results, this.article.title + ".pdf");
                    return;
                }
                var fileURL = URL.createObjectURL(results);
                window.open(fileURL);
            },
            error => console.log(error)
            );
    }

}